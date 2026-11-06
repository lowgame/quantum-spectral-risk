#!/usr/bin/env python3
"""Independent archive audit for experiment 8C.
No archived Python is imported/executed; no training, new sampling or edits to input.
Dependencies: Python>=3.10, NumPy>=2, pandas, SciPy. Run from the output directory:
 python analyze_mechanism8.py --input /path/spektrum_mekanizma8_rapor.zip --out results
"""
from __future__ import annotations
import argparse, collections, hashlib, io, itertools, json, math, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.special import xlogy, xlog1py
from scipy.stats import binomtest

CD=np.complex128

def require(ok, msg):
    if not ok: raise RuntimeError(msg)

def sha(b): return hashlib.sha256(b).hexdigest()
def js(b): return json.loads(b)
def sym(A): return (A+A.conj().T)/2

def save(p, data):
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def spectral(E,w,omega,periodic=False):
    if periodic:
        return .5/(2*np.pi)*np.sinh(.1)*np.sum(w[:,None]/(np.cosh(.1)-np.cos((omega[None,:]-E[:,None])*.5)),axis=0)
    return np.sum(w[:,None]*.2/(np.pi*((omega[None,:]-E[:,None])**2+.04)),axis=0)

def params(m):
    w=np.array(m['weights']); theta=np.array(m['phases']); A=np.array(m['energy_real'])+1j*np.array(m['energy_imag'])
    return w,theta,A

def latent_spectrum(m,info,omega,periodic=False):
    w,_,A=params(m); K=info['sign']*(info['scale']*A-info['shift']*np.eye(len(w)))
    E,V=eigh(sym(K)); weights=abs(V.conj().T@(info['beta']*np.sqrt(w)))**2
    return spectral(E,weights,omega,periodic)

def geometry(raw):
    q=np.array(raw['q']); keys=raw['keys']; ns=sum(k[0]=='S' for k in keys)
    lags=np.array([k[1] for k in keys[:ns]]); si=np.array([k[-1]=='I' for k in keys[:ns]])
    hi=np.array([k[1] for k in keys[ns:]]); hj=np.array([k[2] for k in keys[ns:]]); him=np.array([k[-1]=='I' for k in keys[ns:]])
    return q,ns,lags,si,hi,hj,him

def forward(m,g):
    q,ns,lags,si,hi,hj,him=g
    w,theta,A=params(m)
    V=np.sqrt(w)[:,None]*np.exp(-1j*theta[:,None]*q[None,:])
    s=np.sum(w[:,None]*np.exp(-1j*theta[:,None]*lags[None,:]),axis=0)
    h=(V.conj().T@A@V)[hi,hj]
    return np.r_[np.where(si,s.imag,s.real),np.where(him,h.imag,h.real)]

def kl(pred,y,n):
    p=np.clip((1+pred)/2,1e-12,1-1e-12); q=np.clip((1+y)/2,0,1)
    return float(np.sum(n*(xlogy(q,q)+xlog1py(1-q,-q)-q*np.log(p)-(1-q)*np.log1p(-p)))/n.sum())

def matrix_data(raw):
    q=np.array(raw['q']); n=np.array(raw['counts_train'])+np.array(raw['counts_validation'])
    y=(np.array(raw['counts_train'])*np.array(raw['observed_train'])+np.array(raw['counts_validation'])*np.array(raw['observed_validation']))/n
    S=np.eye(len(q),dtype=CD); H=np.zeros_like(S)
    for key,v in zip(raw['keys'],y):
        if key[0]=='S':
            d=q[None,:]-q[:,None]; part=v if key[-1]=='R' else 1j*v
            S[d==key[1]]+=part; S[d==-key[1]]+=part.conjugate()
        else:
            i,j=key[1:3]; v=v if key[-1]=='R' else 1j*v
            H[i,j]+=v
            if i!=j: H[j,i]+=v.conjugate()
    return S,H,n,y

def lag_project(H,q):
    d=np.array(q)[None,:]-np.array(q)[:,None]; out=np.empty_like(H)
    for k in np.unique(d): out[d==k]=H[d==k].mean()
    return sym(out)

def cutoff(raw,omega,pool=False):
    S,H,n,_=matrix_data(raw)
    if pool: H=lag_project(H,raw['q'])
    info=raw['info']; T=info['sign']*(info['scale']*H-info['shift']*S)
    R=sym(S[1:,1:]-S[1:,:1]@S[:1,1:]); ev,V=eigh(R)
    threshold=max(max(float(ev.max()),0)*1e-8,2*len(S)/math.sqrt(int(n[0])),1e-14)
    keep=ev>threshold
    C0=np.zeros((len(S),1),dtype=CD);C0[0,0]=1
    if keep.any():
        D=V[:,keep]/np.sqrt(ev[keep]); C=np.c_[C0,np.r_[-S[:1,1:]@D,D]]
    else: C=C0
    Tsmall=sym(C.conj().T@T@C); f=C.conj().T@(info['beta']*S[:,0])
    E,V=eigh(Tsmall); weights=abs(V.conj().T@f)**2
    return spectral(E,weights,omega)

def sector(L,nu,nd):
    return [sum(1<<k for k in a)+sum(1<<(L+k) for k in b) for a in itertools.combinations(range(L),nu) for b in itertools.combinations(range(L),nd)]
def ladder(b,i,create):
    if bool((b>>i)&1)==create:return None
    return b^(1<<i), (-1 if (b&((1<<i)-1)).bit_count()%2 else 1)
def Hsector(L,nu,nd,U):
    bs=sector(L,nu,nd);ix={b:i for i,b in enumerate(bs)}; H=np.zeros((len(bs),len(bs)))
    for j,b in enumerate(bs):
        H[j,j]=U*sum(((b>>i)&1)*((b>>(L+i))&1) for i in range(L))-U*(nu+nd)/2
        for spin in [0,L]:
            for i in range(L-1):
                for p,q in [(spin+i,spin+i+1),(spin+i+1,spin+i)]:
                    c=ladder(b,q,False);a=ladder(c[0],p,True) if c else None
                    if a:H[ix[a[0]],j]-=a[1]*c[1]
    return H,bs

def reference(L,U):
    H,bs=Hsector(L,L//2,L//2,U); E,V=eigh(H);E0=E[0];g=V[:,0];out=[]
    for sign in [1,-1]:
        Hb,bb=Hsector(L,L//2+sign,L//2,U);ix={b:i for i,b in enumerate(bb)};b=np.zeros(len(bb))
        for j,bit in enumerate(bs):
            x=ladder(bit,L//2-1,sign==1)
            if x:b[ix[x[0]]]=x[1]*g[j]
        K=sign*(Hb-E0*np.eye(len(Hb)));energies,Vb=eigh(K);weight=abs(Vb.T@b)**2
        out.append(dict(b=b,basis=bb,K=K,E=energies,w=weight,sign=sign,
                        info=dict(scale=2*(L-1)+U*L/4,shift=E0+U*L/4,sign=sign,beta=float(np.linalg.norm(b)))))
    return E0,out

def product_matrices(L,U,step,br,q):
    bs=br['basis'];ix={b:i for i,b in enumerate(bs)};groups=[[],[]]; onsite=[]
    for spin in [0,L]:
        for i in range(L-1):
            aa=[];bb=[];ss=[]
            for j,bit in enumerate(bs):
                x=ladder(bit,spin+i+1,False);y=ladder(x[0],spin+i,True) if x else None
                if y:aa.append(j);bb.append(ix[y[0]]);ss.append(x[1]*y[1])
            groups[i%2].append((np.array(aa),np.array(bb),np.array(ss)))
    for i in range(L): onsite.append(np.array([j for j,b in enumerate(bs) if ((b>>i)&1) and ((b>>(L+i))&1)],dtype=int))
    def step_fn(v,dt):
        h=br['sign']*dt
        for gi,angle in [(0,-h/2),(1,-h/2),(2,U*h),(1,-h/2),(0,-h/2)]:
            if gi==2:
                for mask in onsite:v[mask]*=np.exp(-1j*angle)
            else:
                for a,b,s in groups[gi]:
                    va=v[a].copy();vb=v[b].copy()
                    v[a]=np.cos(angle)*va-1j*np.sin(angle)*s*vb
                    v[b]=np.cos(angle)*vb-1j*np.sin(angle)*s*va
        return v
    bn=br['b']/np.linalg.norm(br['b']);states={0:bn.astype(CD)}
    for sign in [1,-1]:
        v=bn.astype(CD)
        for j in range(1,max(abs(np.array(q)))+1):
            for _ in range(round(.5/step)):v=step_fn(v,sign*step)
            states[sign*j]=v.copy()
    P=np.stack([states[j] for j in q],axis=1); S=P.conj().T@P
    H=br['sign']*br['K']+br['info']['shift']*np.eye(len(br['b']))
    M=P.conj().T@H@P/br['info']['scale']
    return S,M

def measurements(S,M,raw):
    q=raw['q'];ans=[]
    for k in raw['keys']:
        if k[0]=='S':i,j=next((i,j) for i in range(len(q)) for j in range(len(q)) if q[j]-q[i]==k[1]);v=S[i,j]
        else:v=M[k[1],k[2]]
        ans.append(v.real if k[-1]=='R' else v.imag)
    return np.array(ans)

def holm(pvalues):
    p=np.asarray(pvalues); order=np.argsort(p);res=np.ones(len(p));run=0.
    for k,j in enumerate(order):run=max(run,(len(p)-k)*p[j]);res[j]=min(1.,run)
    return res

def analysis(archive,out):
    out.mkdir(parents=True,exist_ok=True); z=zipfile.ZipFile(archive)
    require(z.testzip() is None,'CRC');require(len(z.namelist())==len(set(z.namelist())),'duplicate archive names')
    read=lambda n:js(z.read(n))
    cfg=read('protocol.json'); rev=read('evaluation_revision.json');summary=read('summary.json')
    audit={'input_sha256':sha(archive.read_bytes()),'archive_crc_pass':True,'archived_code_executed':False,
           'new_training':False,'new_measurement_draws':False,'reference_recomputed':True,'summary':summary}
    body={k:v for k,v in cfg.items() if k!='protocol_sha256'}
    require(sha(json.dumps(body,sort_keys=True).encode())==cfg['protocol_sha256'],'protocol hash')
    for key,name in [('code_sha256','spektrum_mekanizma8.py'),('core_sha256','core6.py'),('rival_sha256','spektrum_rakip8b.py')]:require(sha(z.read(name))==cfg[key],name+' hash')
    require(sha(z.read('spektrum_devam8_v2.py'))==rev['runner_sha256'],'revision source hash')
    for name,h in rev['preserved_input_sha256'].items():require(sha(z.read(name))==h,'preserved input changed '+name)
    audit['preserved_file_count']=len(rev['preserved_input_sha256']); audit['protocol_hash']=cfg['protocol_sha256']
    df=pd.read_csv(z.open('metrics.csv'));require(len(df)==800,'metric count')
    require(not df.duplicated(['step','budget','seed','method']).any(),'duplicate metrics')
    require(df.actual_shots.le(df.budget).all(),'over budget')
    require(df.selection_no_spectral_truth.all() and not df.refit_after_selection.any(),'selection flags')
    methods=['raw_cutoff','Hlag_cutoff','diagonal_cv','general_cv_including_diagonal','published_time_central_extension']
    require(set(df.method)==set(methods),'method list')
    for m in methods:
        rr=df[df.method==m]
        require(rr.nonperiodic_relative_L1.isna().all() if m==methods[-1] else rr.nonperiodic_relative_L1.notna().all(),'unexpected missing nonperiodic')
        require(rr.periodic_L1.isna().all() if m in methods[:2] else rr.periodic_L1.notna().all(),'unexpected missing periodic')
    errors=collections.defaultdict(float)
    omega=np.linspace(-12,12,2401); wp=np.linspace(-2*np.pi,2*np.pi,2401)
    E0,branches=reference(cfg['L'],cfg['U'])
    truth=sum(spectral(br['E'],br['w'],omega) for br in branches)
    ptruth=sum(spectral(br['E'],br['w'],wp,True) for br in branches);mass=float(np.trapezoid(truth,omega))
    audit['physical_reference']={'E0':float(E0),'window_mass':mass,'period_mass':float(np.trapezoid(ptruth,wp))}
    # Exact references are evaluator-only: not passed to training or selection checks.
    true_mu={};fitrows=[];selectedrows=[];physical_mse=[];cases=[];raw_count=0
    for step,B,seed in itertools.product(cfg['steps'],cfg['budgets'],cfg['seeds']):
        tag=f'L6_U7_step{step:g}_B{B}_seed{seed}'; cond=read('conditions/'+tag+'.json')
        require(cond['protocol_sha256']==cfg['protocol_sha256'],'condition protocol')
        require(cond['evaluation_revision']==rev['revision'] and cond['evaluation_runner_sha256']==rev['runner_sha256'],'mixed revision')
        cc=pd.read_csv(z.open('curves/'+tag+'.csv'))
        errors['frequency_grid']=max(errors['frequency_grid'],float(abs(cc.omega.to_numpy()-omega).max()))
        errors['physical_reference_curve']=max(errors['physical_reference_curve'],float(abs(cc.exact.to_numpy()-truth).max()))
        rr=df[(df.step==step)&(df.budget==B)&(df.seed==seed)].set_index('method')
        require(set(r['method'] for r in cond['rows'])==set(methods),'condition methods')
        for r in cond['rows']:
            for met in ['nonperiodic_relative_L1','periodic_L1']:
                if r[met] is not None:errors['condition_metric_csv']=max(errors['condition_metric_csv'],abs(r[met]-rr.loc[r['method'],met]))
        for m in methods[:-1]:
            err=float(np.trapezoid(abs(cc[m].to_numpy()-cc.exact.to_numpy()),omega)/np.trapezoid(cc.exact.to_numpy(),omega))
            errors['curve_integrated_metric']=max(errors['curve_integrated_metric'],abs(err-rr.loc[m,'nonperiodic_relative_L1']))
            require(np.isfinite(cc[m]).all() and cc[m].min()>-1e-8,'nonphysical curve')
        recomputed={m:np.zeros_like(omega) for m in methods[:-1]};per={m:np.zeros_like(wp) for m in methods[2:4]};records_count=[]
        for bid,br in enumerate(branches):
            raw=read(f'raw/{tag}_branch{bid}.json');raw_count+=1
            require(raw['protocol_sha256']==cfg['protocol_sha256'],'raw protocol')
            rawhash=sha(json.dumps(raw,sort_keys=True).encode());g=geometry(raw)
            ct=np.array(raw['counts_train']);cv=np.array(raw['counts_validation']);yt=np.array(raw['observed_train']);yv=np.array(raw['observed_validation']);n=ct+cv
            require(len(n)==622 and np.all(n==B//1244),'raw count')
            require(np.array_equal(ct,np.floor(n*.8).astype(int)),'train fraction')
            for y,count in [(yt,ct),(yv,cv)]:
                require(np.isfinite(y).all() and max(abs(y))<=1,'invalid outcome')
                hits=(1+y)*count/2;errors['integer_count_roundoff']=max(errors['integer_count_roundoff'],float(abs(hits-np.rint(hits)).max()))
            for k in ['scale','shift','sign','beta']:errors['source_metadata']=max(errors['source_metadata'],abs(raw['info'][k]-br['info'][k]))
            if (step,bid) not in true_mu:
                S,M=product_matrices(cfg['L'],cfg['U'],step,br,raw['q']);true_mu[step,bid]=measurements(S,M,raw)
            yy=(ct*yt+cv*yv)/n
            # Independent Gram solver for both controls, sharing the original count data.
            recomputed['raw_cutoff']+=cutoff(raw,omega,False);recomputed['Hlag_cutoff']+=cutoff(raw,omega,True)
            diag_candidates=[]; general_candidates=[]
            for rank,start in itertools.product(cfg['ranks'],range(cfg['starts'])):
                dd=[];gg=[]
                for family,dest in [('diag',dd),('general',gg)]:
                    name=f'{family}_r{rank}_s{start}';fpath=f'fits/{tag}_branch{bid}_{name}.json';fit=read(fpath)
                    require(fit['protocol_sha256']==cfg['protocol_sha256'] and fit['raw_sha256']==rawhash,'fit provenance')
                    require([r['nominal_iterations'] for r in fit['records']]==cfg['milestones'],'milestones')
                    for rec in fit['records']:
                        require(rec['name']==name and rec['rank']==rank,'record identity')
                        w,theta,A=params(rec['model']); require(abs(w.sum()-1)<1e-10 and w.min()>0,'weights')
                        require(np.max(abs(A-A.conj().T))<1e-10,'Hermitian')
                        require(np.max(abs(eigh(sym(A),eigvals_only=True)))<1+1e-9,'energy bound')
                        pred=forward(rec['model'],g);require(np.max(abs(pred))<1+1e-9,'invalid probability')
                        tr=kl(pred,yt,ct);vl=kl(pred,yv,cv)
                        errors['training_deviance']=max(errors['training_deviance'],abs(tr-rec['training_deviance']))
                        errors['selection_deviance']=max(errors['selection_deviance'],abs(vl-rec['selection_deviance']))
                        a=rec['fit']
                        fitrows.append(dict(step=step,budget=B,seed=seed,branch=bid,family=family,rank=rank,start=start,milestone=rec['nominal_iterations'],training_deviance=tr,selection_deviance=vl,iterations=a.get('iterations',0),cap_reached=a.get('iteration_cap_reached',False),seconds=a.get('seconds',0),gradient=a.get('fresh_scaled_gradient_max',None)))
                    dest.extend(fit['records'])
                # General initialization embeds best training diagonal, independent of evaluation truth.
                best_train=min(dd,key=lambda x:x['training_deviance'])
                errors['nested_embedding_forward']=max(errors['nested_embedding_forward'],float(abs(forward(best_train['model'],g)-forward(gg[0]['model'],g)).max()))
                diag_candidates.extend(dd);general_candidates.extend([dict(r,name='included_'+r['name']) for r in dd]);general_candidates.extend(gg)
            chosen=cond['selected'][bid]
            for method,candidates in [('diagonal_cv',diag_candidates),('general_cv_including_diagonal',general_candidates)]:
                winner=min(candidates,key=lambda r:(r['selection_deviance'],r['rank'],r['nominal_iterations'],r['name']))
                sel=chosen[method];require(winner==sel,'wrong held-out selection')
                recomputed[method]+=latent_spectrum(sel['model'],raw['info'],omega)
                per[method]+=latent_spectrum(sel['model'],raw['info'],wp,True)
                pred=forward(sel['model'],g);mu=true_mu[step,bid];w,t,A=params(sel['model']);offdiag=float(np.linalg.norm(A-np.diag(np.diag(A))))
                selectedrows.append(dict(step=step,budget=B,seed=seed,branch=bid,method=method,name=sel['name'],rank=sel['rank'],nominal_iterations=sel['nominal_iterations'],offdiagonal_norm=offdiag,effectively_diagonal=offdiag<1e-10,selection_deviance=sel['selection_deviance'],training_deviance=sel['training_deviance'],true_mean_mse=float(np.mean((pred-mu)**2)),true_measurement_KL=kl(pred,mu,n)))
            physical_mse.append(dict(step=step,budget=B,seed=seed,branch=bid,raw_mean_mse=float(np.mean((yy-true_mu[step,bid])**2))))
        for m,spec in recomputed.items():errors['recomputed_spectrum']=max(errors['recomputed_spectrum'],float(abs(spec-cc[m].to_numpy()).max()))
        for m,pspec in per.items():errors['latent_periodic_metric']=max(errors['latent_periodic_metric'],abs(float(np.trapezoid(abs(pspec-ptruth),wp))-rr.loc[m,'periodic_L1']))
        records_count.append(tag)
        if len(fitrows)%2400==0:print('Audited conditions',len(fitrows)//120,flush=True)
    audit.update(raw_files=raw_count,fit_files=len(fitrows)//5,candidate_records=len(fitrows),selected_records=len(selectedrows),max_errors=dict(errors))
    for name in ['condition_metric_csv','curve_integrated_metric','physical_reference_curve','source_metadata','training_deviance','selection_deviance','nested_embedding_forward']: require(errors[name]<1e-8,'audit '+name)
    require(errors['recomputed_spectrum']<1e-6,'spectral mismatch');require(errors['latent_periodic_metric']<1e-6,'periodic mismatch')
    fd=pd.DataFrame(fitrows);sd=pd.DataFrame(selectedrows);sd.to_csv(out/'selected_models.csv',index=False);fd.to_csv(out/'fit_audit.csv',index=False)
    pd.DataFrame(physical_mse).to_csv(out/'raw_error_diagnostics.csv',index=False)
    # Report both axes separately. NaNs here are inapplicable quantities, not solver failures.
    summ=[]
    for met in ['nonperiodic_relative_L1','periodic_L1']:
        for (step,B,m),grp in df.dropna(subset=[met]).groupby(['step','budget','method']):
            a=grp[met].to_numpy();summ.append(dict(metric=met,step=step,budget=B,method=m,n=len(a),median=float(np.median(a)),mean=float(a.mean()),p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max())))
    pd.DataFrame(summ).to_csv(out/'method_summary.csv',index=False)
    # Predefined within this post-hoc analysis: four comparisons x eight cells = 32 tests.
    piv=df.pivot(index=['step','budget','seed'],columns='method',values='nonperiodic_relative_L1')
    rng=np.random.default_rng(20261105); draws=rng.integers(0,20,(100000,20))
    comp=[]
    pairs=[('diagonal_cv','raw_cutoff'),('diagonal_cv','Hlag_cutoff'),('general_cv_including_diagonal','diagonal_cv'),('general_cv_including_diagonal','raw_cutoff')]
    for (step,B),grp in piv.groupby(level=[0,1]):
        grp=grp.sort_index();require(len(grp)==20,'unequal group')
        for a,b in pairs:
            x=grp[a].to_numpy();y=grp[b].to_numpy();delta=y-x;tol=1e-10
            wins=int(sum(delta>tol));losses=int(sum(delta<-tol));ties=20-wins-losses
            p=binomtest(wins,wins+losses,p=.5).pvalue if wins+losses else 1.
            boot=np.median(y[draws],axis=1)-np.median(x[draws],axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            comp.append(dict(step=step,budget=B,method=a,control=b,wins=wins,losses=losses,ties=ties,median_method=float(np.median(x)),median_control=float(np.median(y)),median_difference=float(np.median(y)-np.median(x)),ci_low=float(lo),ci_high=float(hi),paired_mean_difference=float(delta.mean()),p_sign=float(p)))
    c=pd.DataFrame(comp);c['p_holm32']=holm(c.p_sign.to_numpy());c.to_csv(out/'paired_comparisons.csv',index=False)
    counts=sd[sd.method=='general_cv_including_diagonal'].groupby(['step','budget']).agg(branches=('seed','size'),effective_diagonal=('effectively_diagonal','sum'),max_offdiagonal=('offdiagonal_norm','max'),median_true_mse=('true_mean_mse','median'))
    counts['released']=counts.branches-counts.effective_diagonal;counts.reset_index().to_csv(out/'selection_summary.csv',index=False)
    # Retain the initial repair evidence and distinguish it from independent checks above.
    preflights=[n for n in z.namelist() if n.startswith('numerical_audits/preflight')]
    require(len(preflights)==1,'preflight count');pf=read(preflights[0]);save(out/'archived_preflight.json',pf)
    extnames=[n for n in z.namelist() if n.startswith('numerical_audits/extension_')]
    require(len(extnames)==320,'extension audits count')
    ext=[read(n) for n in extnames]
    audit['archived_extension_checks']={'count':len(ext),'min_spectrum':min(e['minimum_spectrum'] for e in ext),'max_period_mass_error':max(abs(e['period_mass']-1) for e in ext),'max_linear_residual':max(e['audit']['extension']['linear_solve_residual'] for e in ext)}
    audit['archived_repair_preflight']={'two_pole_error':pf['tests']['two_pole_extension_error'],'native_dot_error':pf['arithmetic']['native_dot_max_error'],'old_fit_files':pf['saved_fit_audit']['files'],'old_fit_record_count':pf['saved_fit_audit']['model_records']}
    audit['training_blocks']=int((fd.milestone>0).sum());audit['blocks_at_cap']=int(fd.loc[fd.milestone>0,'cap_reached'].sum())
    audit['training_seconds_recorded']=float(fd.seconds.sum());audit['status']='PASS'
    save(out/'audit.json',audit);save(out/'protocol.json',cfg);df.to_csv(out/'original_metrics.csv',index=False)
    # Exact ground/source references saved for independent periodic-competitor reanalysis.
    np.savez(out/'reference_arrays.npz',omega=omega,truth=truth,omega_period=wp,truth_period=ptruth)
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analysis(a.input,a.out)
