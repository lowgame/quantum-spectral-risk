#!/usr/bin/env python3
"""Post-hoc, no-new-shots pilot of an output-oriented pooling diagnostic.

Reads recorded train/selection binomial summaries. Candidates and their
Jacobian use TRAINING data only. Separate recorded selection data score a
frozen first-order target. The score is NOT an unbiased estimator of true
nonlinear/physical spectral risk. The Hoeffding bound covers ONLY its frozen
linear target, conditionally on training. No model training, no archive code
execution, and no edits to the input archive.

Requires NumPy and pandas. Python >= 3.10. Default fixed eta=.2, scoring grid
241, evaluation grid 2401; no tuning to target spectra. All settings shared by
both spectral branches contribute to the combined-spectrum score.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
TRAP = getattr(np, 'trapezoid', np.trapz)


def geometry(keys, q):
    m=len(q); dd=np.asarray(q)[None,:]-np.asarray(q)[:,None]
    ns=sum(k[0]=='S' for k in keys)
    E=np.zeros((len(keys),m,m),complex)
    for j,k in enumerate(keys):
        v=1 if k[-1]=='R' else 1j
        if k[0]=='S':
            E[j,dd==k[1]]=v; E[j,dd==-k[1]]=v.conjugate() if isinstance(v,complex) else v
        else:
            i,l=k[1:3]; E[j,i,l]=v;E[j,l,i]=np.conj(v)
    # Projection in independent-real-coordinate geometry, not full complex Frobenius geometry.
    groups={}
    for j,k in enumerate(keys):
        if k[0]=='S':continue
        d=q[k[2]]-q[k[1]]
        sign=1 if k[-1]=='R' or d>=0 else -1
        groups.setdefault((abs(d),k[-1]),[]).append((j,sign))
    return E,ns,list(groups.values())


def pool(y,groups):
    out=y.copy()
    for g in groups:
        ii=np.array([i for i,_ in g]);ss=np.array([s for _,s in g])
        out[ii]=ss*np.mean(ss*y[ii])
    return out


def matrices(y,E,ns,info):
    S=np.eye(E.shape[-1],dtype=complex)+np.einsum('k,kij->ij',y[:ns],E[:ns])
    M=np.einsum('k,kij->ij',y[ns:],E[ns:])
    T=info['sign']*(info['scale']*M-info['shift']*S)
    return S,T


def whitening(S,n):
    m=len(S);R=S[1:,1:]-S[1:,:1]@S[:1,1:]
    e,V=np.linalg.eigh((R+R.conj().T)/2)
    cut=max(float(e[-1])*1e-8,2*m/math.sqrt(n),1e-14);keep=e>cut
    C=np.eye(m,1,dtype=complex)
    if keep.any():
        D=V[:,keep]/np.sqrt(e[keep])[None,:]
        C=np.column_stack((C,np.vstack((-S[:1,1:]@D,D))))
    assert np.max(abs(C.conj().T@S@C-np.eye(C.shape[1])))<1e-6
    return C


def spec(S,T,C,beta,om,eta=.2):
    h=C.conj().T@T@C; f=C.conj().T@(beta*S[:,0])
    e,V=np.linalg.eigh((h+h.conj().T)/2); a=V.conj().T@f
    return (abs(a[:,None])**2*eta/(np.pi*((om[None,:]-e[:,None])**2+eta**2))).sum(0)


def jacobian(S,T,C,E,ns,info,om,eta=.2):
    h=C.conj().T@T@C; f=C.conj().T@(info['beta']*S[:,0])
    e,V=np.linalg.eigh((h+h.conj().T)/2); a=V.conj().T@f; CV=C@V
    z=om+1j*eta; den=z[None,:]-e[:,None]
    x=CV@(a[:,None]/den);y=CV.conj()@(a.conj()[:,None]/den)
    # y contains entries of the LEFT row, not the conjugate of x.
    ES=E[:ns]; EH=E[ns:]
    s=info['beta']*(ES[:,0,:]@x+ES[:,:,0]@y)
    s-=(z+info['sign']*info['shift'])[None,:]*np.einsum('iw,kij,jw->kw',y,ES,x,optimize=True)
    h=info['sign']*info['scale']*np.einsum('iw,kij,jw->kw',y,EH,x,optimize=True)
    return -np.concatenate((s,h)).imag/math.pi


def frozen_general_spectrum(y,C,E,ns,info,om):
    S,T=matrices(y,E,ns,info); sp=C.conj().T@S@C;tp=C.conj().T@T@C
    f=C.conj().T@(info['beta']*S[:,0]);z=om+.2j
    x=np.linalg.solve(z[:,None,None]*sp[None,:,:]-tp[None,:,:],np.broadcast_to(f,(len(z),len(f)))[:,:,None])[:,:,0]
    return -(x*f.conj()[None,:]).sum(1).imag/math.pi


def check_gradient(raw):
    y=np.array(raw['observed_train']);n=np.array(raw['counts_train']);info=raw['info']
    E,ns,groups=geometry(raw['keys'],raw['q']);S,T=matrices(y,E,ns,info);C=whitening(S,n[0]);om=np.linspace(-8,8,81)
    J=jacobian(S,T,C,E,ns,info,om);errs=[]
    for k in [0,1,ns-1,ns,ns+1,ns+2,len(y)-1]:
        dy=np.zeros_like(y);dy[k]=1e-7
        fd=(frozen_general_spectrum(y+dy,C,E,ns,info,om)-frozen_general_spectrum(y-dy,C,E,ns,info,om))/(2e-7)
        errs.append(float(np.linalg.norm(fd-J[k])/max(np.linalg.norm(fd),1e-8)))
    assert max(errs)<2e-4, errs
    assert np.max(abs(pool(pool(y,groups),groups)-pool(y,groups)))<1e-14
    # Linear-target algebra identity checked on arbitrary real matrices.
    rng=np.random.default_rng(5021);j=rng.normal(size=(7,5));mu=rng.normal(size=5);x=rng.normal(size=5);u=rng.normal(size=7);v=rng.normal(size=7)
    delta=v-u
    lhs=np.linalg.norm(u-j@mu)**2-np.linalg.norm(v-j@mu)**2
    rhs=2*delta@(j@mu-u)-np.dot(delta,delta)
    assert abs(lhs-rhs)<1e-10
    return {'frozen_jacobian_max_relative_fd_error':max(errs),'linear_target_identity_error':abs(lhs-rhs)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    cfg={'eta':.2,'scoring_grid':[-12,12,241],'evaluation_grid':[-12,12,2401],
         'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
         'alpha':.05,'direction':'raw versus lag-pooled ONLY',
         'split':'unchanged recorded 80% training;20% selection; no refit',
         'candidate_fit_and_derivative':'training only',
         'choices':['score>0','score-Hoeffding_radius>0'],
         'test_status':'post-hoc exploratory on already inspected L8 data, not held-out validation',
         'new_shots':0,'new_model_fits':0,'derivative':'full S and H, correlated f, frozen rank',
         'scope':'linearized projected-spectrum risk, not exact physical L1/L2 risk'}
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with zipfile.ZipFile(args.input) as zf:
        cases=sorted(n for n in zf.namelist() if n.startswith('conditions/') and n.endswith('.json'))
        first=cases[0].split('/')[-1][:-5]
        tests=check_gradient(json.loads(zf.read('raw/'+first+'_branch0.json')))
        (out/'self_tests.json').write_text(json.dumps(tests,indent=2)+'\n')
        print('TESTS',tests,flush=True)
        om=np.linspace(-12,12,241);weval=np.linspace(-12,12,2401)
        rows=[]
        for ci,name in enumerate(cases,1):
            tag=name.split('/')[-1][:-5];mat=re.search(r'step([0-9.]+)_B([0-9]+)_seed([0-9]+)',tag)
            step,B,seed=float(mat[1]),int(mat[2]),int(mat[3])
            rawspec=np.zeros_like(om);poolspec=np.zeros_like(om)
            evals={k:np.zeros_like(weval) for k in ('raw_train','pool_train','raw_full','pool_full')}
            work=[];ranks=[]
            for bid in (0,1):
                r=json.loads(zf.read('raw/'+tag+f'_branch{bid}.json'))
                yt=np.array(r['observed_train']);yv=np.array(r['observed_validation']);nt=np.array(r['counts_train']);nv=np.array(r['counts_validation']);info=r['info']
                E,ns,groups=geometry(r['keys'],r['q']);S,T=matrices(yt,E,ns,info);_,Tp=matrices(pool(yt,groups),E,ns,info)
                C=whitening(S,nt[0]);ranks.append(C.shape[1]);J=jacobian(S,T,C,E,ns,info,om)
                rawspec+=spec(S,T,C,info['beta'],om);poolspec+=spec(S,Tp,C,info['beta'],om)
                evals['raw_train']+=spec(S,T,C,info['beta'],weval);evals['pool_train']+=spec(S,Tp,C,info['beta'],weval)
                y=(nt*yt+nv*yv)/(nt+nv);SF,TF=matrices(y,E,ns,info);_,TPF=matrices(pool(y,groups),E,ns,info);CF=whitening(SF,int((nt+nv)[0]))
                evals['raw_full']+=spec(SF,TF,CF,info['beta'],weval);evals['pool_full']+=spec(SF,TPF,CF,info['beta'],weval)
                vhat=(1-yv*yv)/(nv-1)
                work.append((J,yv-yt,nv,vhat))
            delta=poolspec-rawspec
            score=-float(TRAP(delta**2,om));gvar=0.;g_bound=0.
            for J,dy,nv,v in work:
                g=TRAP(J*delta[None,:],om,axis=1)
                score+=2*float(g@dy);gvar+=4*float((g*g)@v);g_bound+=float((g*g/nv).sum())
            radius=math.sqrt(8*math.log(1/.05)*g_bound)
            # Output decisions happen before reading ground-truth curve.
            choose=score>0;safe=score>radius
            evals['score_selector']=evals['pool_train'] if choose else evals['raw_train']
            evals['conservative_selector']=evals['pool_train'] if safe else evals['raw_train']
            ref=pd.read_csv(io.BytesIO(zf.read('curves/'+tag+'.csv')))
            assert np.max(abs(ref['omega'].to_numpy()-weval))<1e-10
            truth=ref['exact'].to_numpy();mass=TRAP(truth,weval)
            rr={'step':step,'budget':B,'seed':seed,'score':score,'surrogate_hoeffding_radius':radius,
                'surrogate_sd_estimate':math.sqrt(gvar),'choose_pool':choose,'choose_pool_conservative':safe,
                'rank_add':ranks[0],'rank_remove':ranks[1]}
            for meth,a in evals.items():
                rr[meth+'_L1']=float(TRAP(abs(a-truth),weval)/mass)
                rr[meth+'_L2']=float(TRAP((a-truth)**2,weval))
            rr['actual_pool_advantage_L2']=rr['raw_train_L2']-rr['pool_train_L2']
            rr['actual_pool_advantage_L1']=rr['raw_train_L1']-rr['pool_train_L1']
            rows.append(rr)
            if ci%10==0:print(f'{ci}/{len(cases)}',flush=True)
        df=pd.DataFrame(rows);df.to_csv(out/'paired_results.csv',index=False)
        summary=[]
        for (step,B),d in df.groupby(['step','budget']):
            rec={'step':step,'budget':int(B),'n':len(d),'pool_choices':int(d.choose_pool.sum()),'conservative_pool_choices':int(d.choose_pool_conservative.sum()),
                 'correct_direction_L2':int((d.choose_pool==(d.actual_pool_advantage_L2>0)).sum()),'pool_actually_better_L2':int((d.actual_pool_advantage_L2>0).sum()),
                 'correct_direction_L1':int((d.choose_pool==(d.actual_pool_advantage_L1>0)).sum()),'pool_actually_better_L1':int((d.actual_pool_advantage_L1>0).sum())}
            for meth in ('raw_full','pool_full','raw_train','pool_train','score_selector','conservative_selector'):
                rec[meth+'_median_L1']=float(d[meth+'_L1'].median());rec[meth+'_mean_L2']=float(d[meth+'_L2'].mean())
            summary.append(rec)
        pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
        (out/'audit.json').write_text(json.dumps({'status':'PASS','cases':len(df),'new_shots':0,'new_fits':0,'tests':tests,'limits':cfg},indent=2)+'\n')
        print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':main()
