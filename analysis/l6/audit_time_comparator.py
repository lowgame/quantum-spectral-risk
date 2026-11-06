#!/usr/bin/env python3
"""Independent periodic-spectrum audit: NumPy projection, 70-digit Gaussian
elimination (not Levinson/Schur), recurrence and damped finite Fourier sum.
No archived source is executed; no spectra are clipped. Training is not run.
"""
from __future__ import annotations
import argparse, hashlib, json, math, zipfile
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from analyze_mechanism8 import require,save

def gram(f):
    d=np.arange(len(f))[None,:]-np.arange(len(f))[:,None]
    return np.where(d>=0,f[abs(d)],f[abs(d)].conj())

def proj(A):
    n=len(A); out=np.empty_like(A)
    for d in range(-n+1,n):
        idx=np.where(np.arange(n)[None,:]-np.arange(n)[:,None]==d)
        out[idx]=A[idx].mean()
    np.fill_diagonal(out,1)
    return (out+out.conj().T)/2

def denoise(f):
    X=proj(gram(f));scale=max(1.,np.linalg.norm(X))
    for it in range(1,6001):
        e,V=eigh(X); Y=proj((V*np.maximum(e,0))@V.conj().T)
        change=np.linalg.norm(Y-X)/scale; X=Y
        if change<1e-10 and eigh(X,eigvals_only=True).min()>=-1e-9*scale:break
    else:raise RuntimeError('projection did not converge')
    out=X[0].copy();emin=float(eigh(gram(out),eigvals_only=True).min())
    mix=max(0.,(1e-10-emin)/(1-emin)) if emin<1e-10 else 0.
    out[1:]*=1-mix;out[0]=1
    return out,it

def dc(x):return Decimal.from_float(float(x.real)),Decimal.from_float(float(x.imag))
def add(a,b):return a[0]+b[0],a[1]+b[1]
def sub(a,b):return a[0]-b[0],a[1]-b[1]
def mul(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def conj(a):return a[0],-a[1]
def norm2(a):return a[0]*a[0]+a[1]*a[1]
def div(a,b):
    n=norm2(b); v=mul(a,conj(b));return v[0]/n,v[1]/n

def coefficients(f):
    with localcontext() as ctx:
        ctx.prec=70;p=len(f)-1; ff=[dc(v) for v in f]
        M=[[ff[i-j] if i>=j else conj(ff[j-i]) for j in range(p)]+[ff[i+1]] for i in range(p)]
        for k in range(p):
            pivot=max(range(k,p),key=lambda i:norm2(M[i][k]));M[k],M[pivot]=M[pivot],M[k]
            for i in range(k+1,p):
                c=div(M[i][k],M[k][k])
                for j in range(k+1,p+1):M[i][j]=sub(M[i][j],mul(c,M[k][j]))
                M[i][k]=(Decimal(0),Decimal(0))
        a=[(Decimal(0),Decimal(0))]*p
        for i in range(p-1,-1,-1):
            v=M[i][p]
            for j in range(i+1,p):v=sub(v,mul(M[i][j],a[j]))
            a[i]=div(v,M[i][i])
        return np.array([complex(float(x),float(y)) for x,y in a])

def spectrum(f,basis):
    f,it=denoise(f);a=coefficients(f);p=len(a)
    series=np.zeros(512,dtype=complex);series[:len(f)]=f
    for k in range(len(f),len(series)):
        terms=a*series[k-p:k][::-1]
        series[k]=complex(math.fsum(terms.real),math.fsum(terms.imag))
    require(max(abs(series))<1+1e-6,'recurrence instability')
    out=.5/(2*np.pi)*(1+2*(series[1:]@basis).real)
    require(np.isfinite(out).all() and out.min()>0,'invalid positive spectrum')
    return out,it

def run(archive,refs,out):
    z=zipfile.ZipFile(archive);df=pd.read_csv(z.open('metrics.csv'));cfg=json.loads(z.read('protocol.json'))
    rr=np.load(refs);wp=rr['omega_period'];truth=rr['truth_period']
    powers=np.exp((-.2+1j*wp[None,:])*.5*np.arange(1,512)[:,None])
    records=[];extension_records=[];difference=0.;match=0.;masses=[]
    for step in cfg['steps']:
        for B in cfg['budgets']:
            for seed in cfg['seeds']:
                tag=f'L6_U7_step{step:g}_B{B}_seed{seed}';total=np.zeros_like(wp)
                for bid in (0,1):
                    raw=json.loads(z.read('raw/'+tag+f'_branch{bid}.json')); ct=np.array(raw['counts_train']);cv=np.array(raw['counts_validation'])
                    yy=(ct*np.array(raw['observed_train'])+cv*np.array(raw['observed_validation']))/(ct+cv)
                    f=np.ones(24,dtype=complex)
                    for k,y in zip(raw['keys'],yy):
                        if k[0]!='S':continue
                        if k[-1]=='R':f[k[1]]=complex(y,f[k[1]].imag)
                        else:f[k[1]]=complex(f[k[1]].real,y)
                    info=raw['info'];sgn=info['sign'];E0=info['shift']-7*6/4;gamma=E0+7*(6+sgn)/2
                    f*=np.exp(1j*sgn*gamma*.5*np.arange(24))
                    sp,it=spectrum(f,powers);masses.append(float(np.trapezoid(sp,wp)));total+=info['beta']**2*sp
                    extension_records.append(dict(step=step,budget=B,seed=seed,branch=bid,iterations=it,min_spectrum=float(sp.min()),period_mass=masses[-1]))
                err=float(np.trapezoid(abs(total-truth),wp));old=float(df[(df.step==step)&(df.budget==B)&(df.seed==seed)&(df.method=='published_time_central_extension')].periodic_L1.iloc[0]);delta=abs(err-old);difference=max(delta,difference)
                records.append(dict(step=step,budget=B,seed=seed,recomputed_periodic_L1=err,archived_periodic_L1=old,difference=delta))
            print('Periodic audit',step,B,flush=True)
    require(difference<1e-7,'periodic comparator metric mismatch')
    require(max(abs(np.array(masses)-1))<1e-7,'period mass mismatch')
    result={'status':'PASS','conditions':len(records),'branches':len(extension_records),'independent_algorithm':'NumPy PSD/Toeplitz, Decimal Gaussian elimination, scalar recurrence, damped Fourier sum','archived_code_executed':False,'max_periodic_metric_difference':difference,'max_period_mass_difference':float(max(abs(np.array(masses)-1))),'fixed_eta':.2,'recurrence_length':512,'theoretical_tail_bound_if_positive_measure':float(.5/np.pi*np.exp(-512*.1)/(1-np.exp(-.1))),'min_spectrum':min(x['min_spectrum'] for x in extension_records)}
    save(out/'time_comparator_audit.json',result);pd.DataFrame(records).to_csv(out/'time_comparator_metrics_recomputed.csv',index=False);pd.DataFrame(extension_records).to_csv(out/'time_comparator_extensions_recomputed.csv',index=False);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.input,a.reference,a.out)
