#!/usr/bin/env python3
"""Independent sparse NumPy/SciPy reference for L8. No archived Python executed.
Only deterministic evaluation; no fitting or simulated random measurements.
"""
from __future__ import annotations
import argparse,itertools,json,math,zipfile,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,eye
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal
from analyze_validation9 import mat,spectrum,periodic,save,require

def states(L,nu,nd):
    u=sorted(sum(1<<i for i in c) for c in itertools.combinations(range(L),nu))
    d=sorted(sum(1<<i for i in c) for c in itertools.combinations(range(L),nd))
    return [a+(b<<L) for a in u for b in d]
def op(bits,k,create):
    if bool((bits>>k)&1)==create:return None
    return bits^(1<<k),(-1 if (bits&((1<<k)-1)).bit_count()%2 else 1)
def hopping(bits,a,b):
    t=op(bits,b,False)
    if t is None:return None
    u=op(t[0],a,True)
    return None if u is None else (u[0],u[1]*t[1])
def hamiltonian(L,nu,nd,U):
    bas=states(L,nu,nd);idx={b:j for j,b in enumerate(bas)};rows=[];cols=[];vals=[]
    for j,b in enumerate(bas):
        rows.append(j);cols.append(j);vals.append(U*sum(((b>>i)&1)*((b>>(i+L))&1) for i in range(L))-U*(nu+nd)/2)
        for s in (0,L):
            for i in range(L-1):
                for a,c in ((s+i,s+i+1),(s+i+1,s+i)):
                    out=hopping(b,a,c)
                    if out is not None:rows.append(idx[out[0]]);cols.append(j);vals.append(-out[1])
    H=coo_matrix((vals,(rows,cols)),shape=(len(bas),len(bas))).tocsr()
    require((H-H.T).nnz==0,'Hermiticity');return H,bas

def lanczos(H,b,om,wp,maximum=1056):
    d=len(b);beta0=np.linalg.norm(b);q=b/beta0
    Q=np.empty((d,maximum),order='F');aa=[];bb=[];last=None;checks=[]
    for k in range(maximum):
        Q[:,k]=q;v=H@q;a=float(q@v);aa.append(a);v-=a*q
        if k:v-=bb[k-1]*Q[:,k-1]
        for _ in range(2):v-=Q[:,:k+1]@(Q[:,:k+1].T@v)
        beta=float(np.linalg.norm(v));bb.append(beta);m=k+1
        if m in (480,720,960,1056) or beta<1e-12:
            E,V=eigh_tridiagonal(np.array(aa),np.array(bb[:-1]));w=beta0**2*V[0]**2
            spec=spectrum(E,w,om);pspec=periodic(E,w,wp)
            tail=np.sum((V[0]*V[-1])[:,None]/(om[None,:]+.2j-E[:,None]),axis=0)
            rb=beta0**2*beta*np.max(abs(tail))/.2
            change=float(np.max(abs(spec-last[0]))) if last is not None else None
            pchange=float(np.max(abs(pspec-last[1]))) if last is not None else None
            checks.append({'m':m,'conditional_resolvent_bound':float(rb),'spectral_change':change,'periodic_change':pchange})
            print('reference m',m,'bound',rb,'change',change,flush=True)
            if last is not None and rb<2e-9 and change<1e-9 and pchange<1e-9:
                return spec,pspec,E,w,checks
            last=(spec,pspec)
        if beta<1e-12:return spec,pspec,E,w,checks
        q=v/beta
    require(rb<1e-7 and change<1e-8,'Reference convergence');return spec,pspec,E,w,checks

def timepool(L,U,bas,b,sgn,step):
    idx={v:j for j,v in enumerate(bas)};pairs=[[],[]];onsite=[]
    for spin in (0,L):
        for site in range(L-1):
            ai=[];bi=[];ss=[]
            for j,x in enumerate(bas):
                v=hopping(x,spin+site,spin+site+1)
                if v is not None:ai.append(j);bi.append(idx[v[0]]);ss.append(v[1])
            pairs[site%2].append((np.array(ai),np.array(bi),np.array(ss)))
    for i in range(L):onsite.append(np.array([j for j,x in enumerate(bas) if ((x>>i)&1) and ((x>>(i+L))&1)]))
    def tick(v,delt):
        h=sgn*delt
        for group,ang in ((pairs[0],-h/2),(pairs[1],-h/2),(onsite,U*h),(pairs[1],-h/2),(pairs[0],-h/2)):
            if group is onsite:
                for inds in group:v[inds]*=np.exp(-1j*ang)
            else:
                for a,c,s in group:
                    va=v[a].copy();vc=v[c].copy();v[a]=np.cos(ang)*va-1j*np.sin(ang)*s*vc;v[c]=np.cos(ang)*vc-1j*np.sin(ang)*s*va
        return v
    v0=b.astype(complex)/np.linalg.norm(b);vs={1:v0.copy(),-1:v0.copy()};states0=[v0]
    for k in range(1,13):
        for sign in (1,-1):
            if len(states0)==24:break
            v=vs[sign]
            for _ in range(round(.5/step)):v=tick(v,sign*step)
            vs[sign]=v;states0.append(v.copy())
    return np.column_stack(states0)

def main(inp,out):
    start=time.perf_counter()
    with zipfile.ZipFile(inp) as z:c=json.loads(z.read('reference_cache.json'))
    L,U=8,7.;H,bas=hamiltonian(L,4,4,U);rng=np.random.default_rng(968531)
    e,V=eigsh(H,k=2,which='SA',tol=1e-13,v0=rng.normal(size=len(bas)),ncv=60)
    ii=np.argsort(e);e=e[ii];g=V[:,ii[0]];E0=float(e[0]);res=np.linalg.norm(H@g-E0*g)
    require(abs(E0-c['E0'])<1e-9 and res<1e-9,'Ground mismatch')
    om=np.array(c['omega']);wp=np.array(c['omega_periodic']);truth=0;ptruth=0
    result={'ground_E0':E0,'ground_residual':float(res),'cached_E0_difference':abs(E0-c['E0']),'ground_gap_numerical':float(e[1]-e[0]),'source':'independent sparse generic-fermionic construction; numerical reference, not global certificate','branches':[],'new_random_measurements':0,'new_training':0};arrays={}
    for bid,sign in enumerate((1,-1)):
        Hb,bb=hamiltonian(L,4+sign,4,U);idx={v:j for j,v in enumerate(bb)};b=np.zeros(len(bb))
        for j,x in enumerate(bas):
            t=op(x,3,sign==1)
            if t:b[idx[t[0]]]+=t[1]*g[j]
        K=sign*(Hb-E0*eye(len(bb),format='csr'));sp,pp,E,w,checks=lanczos(K,b,om,wp)
        truth=truth+sp;ptruth=ptruth+pp
        ba={'dimension':len(bb),'source_norm_squared':float(b@b),'response':checks,'steps':{}}
        for step in (.125,.25):
            P=timepool(L,U,bb,b,sign,step);S=P.conj().T@P;T=P.conj().T@(K@P);f=P.conj().T@b
            back=c['backends'][str(step)][bid];info=back['info'];Hn=(sign*T+info['shift']*S)/info['scale']
            q=[0]+[((j+1)//2)*(1 if j%2 else -1) for j in range(1,24)]
            means=[]
            for lag in range(1,24):
                i,j=next((i,j) for i in range(24) for j in range(24) if q[j]-q[i]==lag);means +=[S[i,j].real,S[i,j].imag]
            for i in range(24):
                for j in range(i,24):
                    means.append(Hn[i,j].real)
                    if i!=j:means.append(Hn[i,j].imag)
            me=float(np.max(abs(np.array(means)-back['means'])))
            dif={k:float(np.max(abs(v-mat(back['data'][k])))) for k,v in [('S',S),('T',T),('f',f)]}
            require(max(dif.values())<1e-8 and me<1e-8,'Ideal measurement cache mismatch')
            ba['steps'][str(step)]={'matrix_max_differences':dif,'measurement_max_difference':me}
            arrays[f'means_step{step}_branch{bid}']=np.array(means)
        arrays[f'energies_branch{bid}']=E;arrays[f'weights_branch{bid}']=w;result['branches'].append(ba)
    arrays.update(omega=om,omega_periodic=wp,reference_nonperiodic=truth,reference_periodic=ptruth)
    err=float(np.max(abs(truth-c['reference_nonperiodic'])));perr=float(np.max(abs(ptruth-c['reference_periodic'])))
    require(err<1e-8 and perr<1e-8,'Physical reference curves mismatch');result.update(status='PASS',physical_curve_max_difference=err,periodic_curve_max_difference=perr,seconds=time.perf_counter()-start)
    out.mkdir(exist_ok=True,parents=True);save(out/'independent_physics_audit.json',result);np.savez_compressed(out/'independent_reference.npz',**arrays)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.input,a.out)
