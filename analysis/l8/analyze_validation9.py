#!/usr/bin/env python3
"""Independent audit of experiment 9. Archived Python is never executed.
Python 3.10+, NumPy, SciPy, pandas. Reads archives; writes only a separate directory.
Post-hoc paired statistical analysis; not an independent laboratory replication.
"""
from __future__ import annotations
import argparse, hashlib, io, itertools, json, math, re, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binomtest

TRAP = getattr(np, 'trapezoid', np.trapz)

def load(z,n): return json.loads(z.read(n))
def save(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def h(b): return hashlib.sha256(b).hexdigest()
def require(x,msg):
    if not x: raise RuntimeError(msg)
def mat(v):return np.asarray(v['real'])+1j*np.asarray(v['imag'])
def pars(m):return np.array(m['phases']),np.array(m['weights']),np.array(m['energy_real'])+1j*np.array(m['energy_imag'])

def model_means(m,keys,q):
    theta,w,A=pars(m);V=np.sqrt(w)[:,None]*np.exp(-1j*theta[:,None]*np.array(q)[None,:])
    ss=(w[:,None]*np.exp(-1j*theta[:,None]*np.arange(1,24)[None,:])).sum(0)
    hh=V.conj().T@A@V
    return np.array([(ss[k[1]-1] if k[0]=='S' else hh[k[1],k[2]]).real if k[-1]=='R' else
                     (ss[k[1]-1] if k[0]=='S' else hh[k[1],k[2]]).imag for k in keys])

def dev(pred,obs,counts):
    p=np.clip((1+pred)/2,1e-12,1-1e-12);q=np.clip((1+obs)/2,0,1)
    vals=xlogy(q,q)+xlogy(1-q,1-q)-q*np.log(p)-(1-q)*np.log1p(-p)
    return float(counts@vals/counts.sum())

def spectrum(E,w,om,eta=.2):
    return (w[:,None]*eta/(np.pi*((om[None,:]-E[:,None])**2+eta**2))).sum(0)
def periodic(E,w,om,eta=.2,dt=.5):
    return (dt/(2*np.pi)*np.sinh(eta*dt)*w[:,None]/(np.cosh(eta*dt)-np.cos((om[None,:]-E[:,None])*dt))).sum(0)
def model_poles(model,info):
    _,w,A=pars(model);K=info['sign']*(info['scale']*A-info['shift']*np.eye(len(w)))
    E,V=np.linalg.eigh((K+K.conj().T)/2);f=info['beta']*np.sqrt(w)
    return E, np.abs(V.conj().T@f)**2

def matrices(raw):
    ct=np.array(raw['counts_train']);cv=np.array(raw['counts_validation']);cn=ct+cv
    y=(ct*np.array(raw['observed_train'])+cv*np.array(raw['observed_validation']))/cn
    q=np.array(raw['q']);d=q[None,:]-q[:,None];S=np.eye(len(q),dtype=complex);H=np.zeros_like(S)
    for k,v in zip(raw['keys'],y):
        if k[0]=='S':
            z=v if k[-1]=='R' else 1j*v
            S[d==k[1]]+=z;S[d==-k[1]]+=np.conj(z)
        else:
            i,j=k[1:3];z=v if k[-1]=='R' else 1j*v
            H[i,j]+=z
            if i!=j:H[j,i]+=np.conj(z)
    return S,H,y,cn

def poolH(H,q):
    q=np.array(q);d=q[None,:]-q[:,None];result=np.empty_like(H)
    for lag in np.unique(d):result[d==lag]=H[d==lag].mean()
    return result

def projected(S,H,info,shots):
    R=S[1:,1:]-S[1:,:1]@S[:1,1:];ev,V=np.linalg.eigh((R+R.conj().T)/2)
    threshold=max(float(ev.max())*1e-8,2*len(S)/math.sqrt(shots),1e-14)
    keep=ev>threshold;C0=np.eye(len(S),1,dtype=complex)
    D=V[:,keep]/np.sqrt(ev[keep])[None,:]
    C=np.column_stack((C0,np.vstack((-S[:1,1:]@D,D)))) if keep.any() else C0
    T=info['sign']*(info['scale']*H-info['shift']*S);f=info['beta']*S[:,0]
    Tp=C.conj().T@T@C;fp=C.conj().T@f
    E,U=np.linalg.eigh((Tp+Tp.conj().T)/2)
    return E,np.abs(U.conj().T@fp)**2, {'rank':C.shape[1],'cutoff':threshold}

def best(records):return min(records,key=lambda r:(r['selection_deviance'],r['rank'],r['nominal_iterations'],r['name']))
def maxdiff(a,b): return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))

def analyze(inp,out,boot=100000):
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    audit={'input':inp.name,'input_sha256':h(inp.read_bytes()),'archived_python_executed':False,'new_fits':0,'new_random_measurements':0,'bootstrap_samples':boot}
    errors={}; rows=[]; branchrows=[]; fitrows=[]; selection=[]; measure=[];worst=[]
    def err(k,v):errors[k]=max(errors.get(k,0.),float(v))
    with zipfile.ZipFile(inp) as z:
        names=z.namelist();require(len(names)==len(set(names)),'Duplicate archive members');require(z.testzip() is None,'CRC error')
        cfg=load(z,'protocol.json');ph=cfg['protocol_sha256'];body={k:v for k,v in cfg.items() if k!='protocol_sha256'}
        require(h(json.dumps(body,sort_keys=True).encode())==ph,'Protocol hash');audit['protocol_sha256']=ph
        for name,dig in cfg['hashes'].items():require(h(z.read('source/'+name))==dig,'Source hash '+name)
        summary=load(z,'summary.json');cache=load(z,'reference_cache.json')
        require(summary['status']=='PASS' and summary['conditions_completed']==80,'Incomplete')
        require(cache['protocol_sha256']==ph and summary['protocol_sha256']==ph,'Reference/summary protocol')
        audit['archive_member_counts']={k:sum(n.startswith(k+'/') for n in names) for k in ('conditions','raw','fits','curves')}
        require(audit['archive_member_counts']=={'conditions':80,'raw':160,'fits':1920,'curves':160},'Member counts')
        df=pd.read_csv(io.BytesIO(z.read('metrics.csv'))); require(len(df)==480 and not df.duplicated(['step','budget','seed','method']).any(),'Metrics duplicates')
        om=np.array(cache['omega']);wp=np.array(cache['omega_periodic']);truth=np.array(cache['reference_nonperiodic']);ptruth=np.array(cache['reference_periodic']);mass=TRAP(truth,om)
        tre=sum(spectrum(np.array(b['E']),np.array(b['weights']),om) for b in cache['branches'])
        pre=sum(periodic(np.array(b['E']),np.array(b['weights']),wp) for b in cache['branches'])
        err('cached_poles_to_physical_reference',maxdiff(tre,truth));err('cached_poles_to_periodic_reference',maxdiff(pre,ptruth))
        audit['reference_cache_ground']=cache['ground'];audit['reference_cache_branch_audits']=[b['audit'] for b in cache['branches']]
        audit['physical_window_mass']=float(mass)
        for step,B,seed in itertools.product(cfg['steps'],cfg['budgets'],cfg['seeds']):
            tag=f'L8_U7_step{step:g}_B{B}_seed{seed}';cond=load(z,'conditions/'+tag+'.json');require(cond['protocol_sha256']==ph,'Condition protocol')
            cv=pd.read_csv(io.BytesIO(z.read('curves/'+tag+'.csv')));pv=pd.read_csv(io.BytesIO(z.read('curves/'+tag+'_periodic.csv')))
            err('curve_reference',maxdiff(cv.exact,truth));err('periodic_curve_reference',maxdiff(pv.exact,ptruth))
            err('curve_grid',max(maxdiff(cv.omega,om),maxdiff(pv.omega,wp)))
            specs={k:np.zeros_like(om) for k in ('raw_cutoff','Hlag_cutoff','diagonal_cv','general_cv_including_diagonal')}
            pspec={k:np.zeros_like(wp) for k in ('diagonal_cv','general_cv_including_diagonal')}
            charge=0;cost=0;timecost=0;timeshots=0
            for bid in (0,1):
                raw=load(z,'raw/'+tag+f'_branch{bid}.json');require(raw['protocol_sha256']==ph,'Raw protocol')
                rawhash=h(json.dumps(raw,sort_keys=True).encode());ct=np.array(raw['counts_train']);vv=np.array(raw['counts_validation']);cn=ct+vv
                yt=np.array(raw['observed_train']);yv=np.array(raw['observed_validation']);keys=raw['keys'];info=raw['info'];q=raw['q']
                require(len(keys)==622 and len(q)==24,'Geometry changed');require(np.all(cn==B//1244) and np.array_equal(ct,np.floor(.8*cn).astype(int)),'Split count')
                require(np.max(np.abs(yt))<=1 and np.max(np.abs(yv))<=1,'Invalid measurement')
                for yy,nn in [(yt,ct),(yv,vv)]:err('binomial_count_roundoff',np.max(np.abs(nn*(yy+1)/2-np.round(nn*(yy+1)/2))))
                tick=np.array([abs(k[1]) if k[0]=='S' else abs(q[k[1]])+abs(q[k[2]]) for k in keys]);nr=round(.5/step)*36
                charge+=int(cn.sum());cost+=int(cn@tick)*nr;nt=B//92;timeshots+=46*nt;timecost+=int(tick[:46].sum())*nt*nr
                S,H,y,nn=matrices(raw);Hp=poolH(H,q)
                for name,hh in [('raw_cutoff',H),('Hlag_cutoff',Hp)]:
                    E,w,pa=projected(S,hh,info,int(nn[0]));specs[name]+=spectrum(E,w,om)
                    branchrows.append(dict(step=step,budget=B,seed=seed,branch=bid,method=name,rank=pa['rank'],cutoff=pa['cutoff']))
                true_mu=np.array(cache['backends'][str(step)][bid]['means'])
                diag=[];gen=[]
                for r,s in itertools.product(cfg['ranks'],range(cfg['starts'])):
                    for family in ('diag','general'):
                        name=f'{family}_r{r}_s{s}';fit=load(z,'fits/'+tag+f'_branch{bid}_{name}.json')
                        require(fit['protocol_sha256']==ph and fit['raw_sha256']==rawhash,'Fit provenance')
                        require([rec['nominal_iterations'] for rec in fit['records']]==cfg['milestones'],'Milestones')
                        for rec in fit['records']:
                            pred=model_means(rec['model'],keys,q)
                            err('training_loss',abs(dev(pred,yt,ct)-rec['training_deviance']));err('selection_loss',abs(dev(pred,yv,vv)-rec['selection_deviance']))
                            th,w,A=pars(rec['model']);err('mass_constraint',abs(w.sum()-1));err('hermiticity',maxdiff(A,A.conj().T))
                            require(w.min()>=0 and np.linalg.eigvalsh(A).min()>=-1-1e-9 and np.linalg.eigvalsh(A).max()<=1+1e-9,'Unphysical model')
                            f=rec['fit'];fitrows.append(dict(step=step,budget=B,seed=seed,branch=bid,family=family,rank=r,start=s,milestone=rec['nominal_iterations'],iterations=f.get('iterations',0),iteration_cap_reached=f.get('iteration_cap_reached',False),seconds=f.get('seconds',0),fresh_scaled_gradient_max=f.get('fresh_scaled_gradient_max',None)))
                            (diag if family=='diag' else gen).append(rec)
                gen += [dict(rec,name='included_'+rec['name']) for rec in diag]
                ch=cond['selected'][bid];require(ch['candidate_counts']=={'diagonal':30,'general':60},'Candidate counts')
                require(ch['selection_uses_true_spectrum'] is False and ch['refitted_after_selection'] is False,'Selection contract')
                for family,recs in [('diagonal_cv',diag),('general_cv_including_diagonal',gen)]:
                    winner=best(recs);given=ch[family]; require(winner==given,'Selected snapshot mismatch')
                    E,w=model_poles(given['model'],info);specs[family]+=spectrum(E,w,om);pspec[family]+=periodic(E,w,wp)
                    theta,weights,A=pars(given['model']);off=np.linalg.norm(A-np.diag(np.diag(A)))
                    pred=model_means(given['model'],keys,q)
                    branchrows.append(dict(step=step,budget=B,seed=seed,branch=bid,method=family,rank=given['rank'],milestone=given['nominal_iterations'],name=given['name'],offdiagonal_norm=float(off),released=bool(off>1e-10),selection_deviance=given['selection_deviance'],training_deviance=given['training_deviance'],true_mean_mse=float(np.mean((pred-true_mu)**2)),true_mean_KL=dev(pred,true_mu,cn),source_mass=float(w.sum())))
                mean_pool=y.copy()
                for k,key in enumerate(keys):
                    if key[0]=='H': mean_pool[k]=Hp[key[1],key[2]].real if key[-1]=='R' else Hp[key[1],key[2]].imag
                measure.append(dict(step=step,budget=B,seed=seed,branch=bid,raw_MSE=float(np.mean((y-true_mu)**2)),Hlag_MSE=float(np.mean((mean_pool-true_mu)**2)),raw_H_MSE=float(np.mean((y[46:]-true_mu[46:])**2)),Hlag_H_MSE=float(np.mean((mean_pool[46:]-true_mu[46:])**2))))
            for name,spec in specs.items():err('physical_spectrum_from_raw_or_model',maxdiff(spec,cv[name]))
            for name,spec in pspec.items():err('periodic_model_spectrum',maxdiff(spec,pv[name]))
            for r in cond['rows']:
                same=df[(df.step==step)&(df.budget==B)&(df.seed==seed)&(df.method==r['method'])];require(len(same)==1,'Missing condition row')
                rr=same.iloc[0]
                for metric,tab,grid,den in [('nonperiodic_relative_L1',cv,om,mass),('periodic_L1',pv,wp,1.)]:
                    if r[metric] is not None:
                        calc=float(TRAP(abs(tab[r['method']]-tab.exact),grid)/den)
                        err('metric_from_curves',abs(calc-r[metric]));err('metric_condition_vs_global_csv',abs(rr[metric]-r[metric]))
                    else:require(pd.isna(rr[metric]),'NA mismatch')
                shots=timeshots if r['method']=='time_full_budget' else charge
                rcost=timecost if r['method']=='time_full_budget' else cost
                require(r['actual_shots']==shots<=B and r['shot_weighted_rotation_proxy']==rcost,'Budget/cost mismatch')
                require(rr.actual_shots==shots and rr.shot_weighted_rotation_proxy==rcost,'Global costs mismatch')
                rows.append(r)
            if seed==cfg['seeds'][-1]: print(f'AUDIT step={step} B={B}: 20/20 complete',flush=True)
        require(errors['metric_from_curves']<1e-10 and errors['physical_spectrum_from_raw_or_model']<1e-7 and errors['selection_loss']<1e-12,'Numerical discrepancy')
        audit['status']='PASS';audit['max_absolute_differences']=errors
        audit['conditions']=80;audit['results']=480;audit['model_checkpoints']=len(fitrows);audit['trained_blocks']=sum(f['milestone']>0 for f in fitrows)
        audit['native_cpu_complex_dot_is_bad']=load(z,'self_tests.json')['cpu_arithmetic']['native_dot_max_error']
        audit['limits']=['Source and energy reference numerical, preparation excluded; ideal gates.', 'Full-budget time-only raw resamples are not archived; their spectra and costs, but not their raw probability samples, can be independently audited here.', 'Independent NumPy calculations use the cached numerical reference in this first pass; a separate sparse-Hamiltonian verification is documented separately.', 'Analysis family was specified after seeing these data; no prospective preregistration claim.', 'Frozen estimator settings extend to L8, but pilot L8/seed9100 had been inspected before validation.']
        save(out/'audit.json',audit);save(out/'original_protocol.json',cfg)
        df=pd.DataFrame(rows);bf=pd.DataFrame(branchrows);ff=pd.DataFrame(fitrows);mf=pd.DataFrame(measure)
        df.to_csv(out/'all_metrics.csv',index=False);bf.to_csv(out/'branch_diagnostics.csv',index=False);ff.to_csv(out/'fit_diagnostics.csv',index=False);mf.to_csv(out/'measurement_diagnostics.csv',index=False)
        summaries=[]
        for (step,B,method),g in df.groupby(['step','budget','method']):
            for metric in ('nonperiodic_relative_L1','periodic_L1'):
                v=g[metric].dropna().to_numpy()
                if len(v):summaries.append(dict(step=step,budget=B,method=method,metric=metric,n=len(v),median=float(np.median(v)),mean=float(np.mean(v)),p90=float(np.quantile(v,.9)),maximum=float(v.max()),minimum=float(v.min()),median_rotation_cost=float(g.shot_weighted_rotation_proxy.median())))
        pd.DataFrame(summaries).to_csv(out/'method_summary.csv',index=False)
        for (step,B),g in bf[bf.method=='general_cv_including_diagonal'].groupby(['step','budget']):
            selection.append(dict(step=step,budget=B,branches=len(g),released=int(g.released.sum()),rank14=int((g['rank']==14).sum()),last_milestone=int((g.milestone==1400).sum())))
        pd.DataFrame(selection).to_csv(out/'selection_summary.csv',index=False)
        pairs=[('diagonal_cv','raw_cutoff','nonperiodic_relative_L1'),('diagonal_cv','Hlag_cutoff','nonperiodic_relative_L1'),('general_cv_including_diagonal','diagonal_cv','nonperiodic_relative_L1'),('general_cv_including_diagonal','raw_cutoff','nonperiodic_relative_L1'),('diagonal_cv','time_full_budget','periodic_L1'),('general_cv_including_diagonal','time_full_budget','periodic_L1')]
        rng=np.random.default_rng(20261105);bootids=rng.integers(0,20,(boot,20));comparisons=[]
        for step,B in itertools.product(cfg['steps'],cfg['budgets']):
            sub=df[(df.step==step)&(df.budget==B)]
            for m,control,metric in pairs:
                a=sub[sub.method==m].sort_values('seed')[metric].to_numpy();b=sub[sub.method==control].sort_values('seed')[metric].to_numpy()
                require(len(a)==len(b)==20,'Pair mismatch');delta=b-a;win=int((delta>1e-10).sum());lose=int((delta< -1e-10).sum());ties=20-win-lose
                p=float(binomtest(win,win+lose,.5).pvalue) if win+lose else 1.
                samples=np.median(b[bootids],axis=1)-np.median(a[bootids],axis=1);ci=np.quantile(samples,[.025,.975])
                comparisons.append(dict(step=step,budget=B,method=m,control=control,metric=metric,wins=win,losses=lose,ties=ties,median_difference=float(np.median(b)-np.median(a)),relative_median_reduction=float(1-np.median(a)/np.median(b)),CI95_low=float(ci[0]),CI95_high=float(ci[1]),p_sign=p))
        # One post-hoc family: 6 planned pairwise contrasts x 4 cells = 24 tests.
        ps=np.array([r['p_sign'] for r in comparisons]);order=np.argsort(ps);adj=np.empty_like(ps)
        adj[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))))
        for r,p in zip(comparisons,adj):r['p_holm24_sensitivity']=float(p)
        # Primary family predeclared in the experiment-9 delivery: two contrasts x four cells.
        for flag in (True,False):
            ii=[i for i,r in enumerate(comparisons) if ((r['method']=='diagonal_cv' and r['control'] in ('raw_cutoff','Hlag_cutoff'))==flag)]
            vp=np.array([comparisons[i]['p_sign'] for i in ii]);oo=np.argsort(vp);aa=np.empty_like(vp)
            aa[oo]=np.minimum(1,np.maximum.accumulate(vp[oo]*(len(vp)-np.arange(len(vp)))))
            for i,pp in zip(ii,aa):
                comparisons[i]['family']='predeclared_primary8' if flag else 'posthoc_secondary16'
                comparisons[i]['p_holm_family']=float(pp)

        pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
        print(pd.DataFrame(summaries).query("metric=='nonperiodic_relative_L1'").to_string(index=False),flush=True)
        print(pd.DataFrame(comparisons).to_string(index=False),flush=True)
        audit['audit_seconds']=time.perf_counter()-start;save(out/'audit.json',audit)
        return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=100000);a=p.parse_args();analyze(a.input,a.out,a.bootstrap)
