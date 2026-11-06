#!/usr/bin/env python3
"""Independently reconstruct both time-only controls.
NumPy cyclic projection + 70-digit direct Yule-Walker solve + scalar recurrence.
No archived Python executed. Missing full-budget outcomes are deterministically
regenerated from saved seeds/means (not newly chosen independent test trials).
"""
import argparse,hashlib,io,itertools,json,math,time,zipfile
from pathlib import Path
import numpy as np
import mpmath as mp
import pandas as pd
import torch
from analyze_validation9 import save,require

def clean_moments(f):
    n=len(f);q=np.arange(n);d=q[None,:]-q[:,None]
    A=np.where(d>=0,f[abs(d)],np.conj(f[abs(d)]));X=A.copy();scale=max(1.,np.linalg.norm(X));indices=[d==k for k in range(-n+1,n)]
    for it in range(6000):
        e,V=np.linalg.eigh((X+X.conj().T)/2);pos=(V*np.maximum(e,0))@V.conj().T;new=np.empty_like(X)
        for k,idx in zip(range(-n+1,n),indices):new[idx]=pos[idx].mean() if k else 1.
        change=np.linalg.norm(new-X)/scale;X=(new+new.conj().T)/2
        if change<1e-10 and np.linalg.eigvalsh(X).min()>=-1e-9*scale:break
    else:raise RuntimeError('Independent cyclic projection did not converge')
    return X[0].copy()

def continuation(f,length=512):
    n=len(f);d=np.arange(n)[None,:]-np.arange(n)[:,None]
    A=np.where(d>=0,f[abs(d)],np.conj(f[abs(d)]));emin=float(np.linalg.eigvalsh(A).min())
    target=1e-10*f[0].real;mix=max(0.,(target-emin)/(f[0].real-emin)) if emin<target else 0.
    f=f.copy();f[1:]*=1-mix;p=n-1
    with mp.workdps(70):
        ff=[mp.mpc(float(x.real),float(x.imag)) for x in f]
        R=mp.matrix([[ff[i-j] if i>=j else ff[j-i].conjugate() for j in range(p)] for i in range(p)])
        aa=mp.lu_solve(R,mp.matrix(ff[1:]));co=[complex(v) for v in aa]
    vals=list(f)
    for k in range(n,length):
        terms=[co[j]*vals[k-1-j] for j in range(p)];vals.append(complex(math.fsum(x.real for x in terms),math.fsum(x.imag for x in terms)))
    require(np.max(np.abs(vals))<1.001,'Unbounded extension sequence')
    return np.array(vals)

def main(inp,out):
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();mx={'same_data_curve':0.,'regenerated_full_budget_curve':0.};minimum=1e10;masserr=0.;raws=[]
    with zipfile.ZipFile(inp) as z:
        cfg=json.loads(z.read('protocol.json'));cache=json.loads(z.read('reference_cache.json'));wp=np.array(cache['omega_periodic']);freq=np.exp((-.2+1j*wp[None,:])*.5*np.arange(512)[:,None])
        for step,B,seed in itertools.product(cfg['steps'],cfg['budgets'],cfg['seeds']):
            tag=f'L8_U7_step{step:g}_B{B}_seed{seed}';total={'time_same_data':0.,'time_full_budget':0.}
            for bid in (0,1):
                raw=json.loads(z.read('raw/'+tag+f'_branch{bid}.json'));info=raw['info'];back=cache['backends'][str(step)][bid]
                ct=np.array(raw['counts_train']);cv=np.array(raw['counts_validation']);y=(ct*np.array(raw['observed_train'])+cv*np.array(raw['observed_validation']))/(ct+cv)
                token=f"{back['sampling_label']}|seed={seed}|stage=time_only_full_budget";gen=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'little')%(2**63-1))
                shots=B//92;means=torch.tensor(back['means'][:46],dtype=torch.float64);cn=torch.full((46,),float(shots),dtype=torch.float64)
                full=(2*torch.binomial(cn,(1+means)/2,generator=gen)/shots-1).numpy()
                raws.append({'condition':tag,'branch':bid,'counts_per_setting':shots,'means_full_budget_recreated':full.tolist(),'sampling_seed_label':token})
                for name,obs in [('time_same_data',y[:46]),('time_full_budget',full)]:
                    f=np.ones(24,dtype=complex);f[1:]=obs[0::2]+1j*obs[1::2]
                    gamma=cache['E0']+7*(8+info['sign'])/2
                    f*=np.exp(1j*info['sign']*gamma*.5*np.arange(24));clean=clean_moments(f);ex=continuation(clean)
                    ss=.5/(2*np.pi)*(2*(ex@freq).real-ex[0].real)
                    require(np.isfinite(ss).all() and ss.min()>-1e-9,'Nonpositive independent spectrum')
                    minimum=min(minimum,float(ss.min()));masserr=max(masserr,abs(float(np.trapezoid(ss,wp))-1))
                    total[name]+=info['beta']**2*ss
            curves=pd.read_csv(io.BytesIO(z.read('curves/'+tag+'_periodic.csv')))
            mx['same_data_curve']=max(mx['same_data_curve'],float(np.max(abs(total['time_same_data']-curves.time_same_data))))
            mx['regenerated_full_budget_curve']=max(mx['regenerated_full_budget_curve'],float(np.max(abs(total['time_full_budget']-curves.time_full_budget))))
            if seed==cfg['seeds'][-1]:print('TIME step',step,'B',B,'max',mx,flush=True)
    require(mx['same_data_curve']<1e-7,'Same-data mismatch')
    result={'status':'PASS' if max(mx.values())<1e-7 else 'PARTIAL','max_curve_difference':mx,'minimum_branch_spectrum':minimum,'maximum_branch_period_mass_error':masserr,'local_torch':str(torch.__version__),'modes_evaluated':512,'direct_solve_decimal_digits':70,'full_budget_original_raw_missing':True,'full_budget_outcomes_recreated_from_seeds':True,'new_independent_measurement_trials':0,'seconds':time.perf_counter()-start,'scope':'NumPy projection + mpmath direct solve + scalar series, unlike the archived Levinson/Schur evaluator. Full-budget measurements were not stored, but reconstructed deterministic local binomial draws reproduce saved spectra when error <1e-7. This is not recovery of a contemporaneous original raw record.'}
    save(out/'time_comparator_audit.json',result);save(out/'time_full_budget_reconstructed_counts.json',raws);print(result,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.input,a.out)
