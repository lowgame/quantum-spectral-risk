#!/usr/bin/env python3
"""Experiment 8B: real-time PSD denoising -> positive extension -> spectrum.

Kemper, Yang, Gull, Phys. Rev. Lett. 132, 160403 (2024), arXiv:2309.02566v2.
The PSD projection and one-point extension constraints are from that paper.
When extensions are nonunique, this implementation chooses the Schur-disk
center (maximum-entropy autoregressive continuation). This is an explicit
implementation choice, NOT a claim to reproduce the authors' numerical data.
Exponential broadening eta=.2 is fixed; its Fourier transform is evaluated
analytically. No free fitted peak widths and no true-spectrum model selection.

IMPORTANT: sampled-time spectra are periodic (aliasing). We evaluate on ONE
period and distinguish the physical-H spectrum from the product-unitary phase
spectrum. These L1 errors are NOT the previous [-12,12] nonperiodic L1 metric.

Tracks: (1) SAME archived data, only S used; all archived acquisition charged.
(2) NEW classical binomial simulation, entire same shot budget spent on S.
Source/E0 ideal in both; no device noise, native gate or total-runtime claim.
The two tracks are distinct and individually labelled in every result.

CPU only, Python 3.10+, PyTorch. Originals are read, not overwritten/executed.
Uses checkpoints, protocol hashing, and a process lock for safe resumption.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, platform, statistics, sys, time, traceback, zipfile
from pathlib import Path
from typing import Any
from itertools import combinations
import torch
CDTYPE=torch.complex128
RDTYPE=torch.float64
VERSION='0.8b.1-central-extension-periodic-audit'
ETA=.2
SPACING=.5
SOURCE_URL='https://arxiv.org/html/2309.02566v2'

_LAG_MAPS={}
def integral(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sum((y[..., 1:] + y[..., :-1]) * (x[1:] - x[:-1]) * 0.5, dim=-1)


def dump_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def hermitian(A):
    return (A+A.mH)/2


def lag_projection(A:torch.Tensor,q:list[int],diag:float|None=None)->torch.Tensor:
    """Orthogonal Toeplitz projection in a possibly permuted integer time order."""
    key=(tuple(q),str(A.device))
    if key not in _LAG_MAPS:
        qt=torch.tensor(q,dtype=torch.long,device=A.device)
        d=qt[None,:]-qt[:,None];lo=int(d.min())
        ids=(d-lo).reshape(-1)
        counts=torch.bincount(ids).to(RDTYPE)
        _LAG_MAPS[key]=(ids,counts,-lo)
    ids,counts,zero=_LAG_MAPS[key]
    sums=torch.zeros(len(counts),dtype=CDTYPE,device=A.device)
    sums.scatter_add_(0,ids,A.reshape(-1))
    means=sums/counts.clamp_min(1)
    if diag is not None: means[zero]=complex(diag,0)
    return hermitian(means[ids].reshape(A.shape))


def published_denoise(A:torch.Tensor,q:list[int],norm:float=1.,
                      tolerance:float=1e-10,max_iter:int=6000)->tuple[torch.Tensor,dict]:
    """Main-text cyclic projections, NOT claimed to be the global nearest point."""
    check(norm>0 and math.isfinite(norm),'Invalid equal-time normalization.')
    X=lag_projection(hermitian(A),q,norm)
    scale=max(1.,float(torch.linalg.matrix_norm(X)))
    for it in range(1,max_iter+1):
        vals,V=torch.linalg.eigh(X)
        psd=(V*vals.clamp_min(0)[None,:])@V.mH
        new=lag_projection(psd,q,norm)
        change=float(torch.linalg.matrix_norm(new-X))/scale
        X=new
        if change<tolerance:
            mineig=float(torch.linalg.eigvalsh(X).min())
            if mineig>=-10*tolerance*scale: break
    else: raise RuntimeError('PSD/Toeplitz projections did not converge; no silent success.')
    mineig=float(torch.linalg.eigvalsh(X).min())
    toeplitz_error=float((lag_projection(X,q,norm)-X).abs().max())
    check(mineig>=-10*tolerance*scale,'Final matrix not PSD within tolerance.')
    return X,{'iterations':it,'relative_change':change,'minimum_eigenvalue':mineig,
              'toeplitz_residual':toeplitz_error,'norm':norm,
              'global_nearest_point_claimed':False}


def annihilation(orbital:int,norb:int=4)->torch.Tensor:
    dim=2**norb; c=torch.zeros((dim,dim),dtype=CDTYPE)
    for bits in range(dim):
        if (bits>>orbital)&1:
            sign=-1 if (bits&((1<<orbital)-1)).bit_count()%2 else 1
            c[bits^(1<<orbital),bits]=sign
    return c


def paper_dimer():
    """Independent full-Fock implementation of source Appendix C, Eqs.12-15."""
    c=[annihilation(j) for j in range(4)]
    n=[a.mH@a for a in c]
    H=-2.3*sum(n)
    for p,q in ((0,1),(2,3)): H=H-(c[p].mH@c[q]+c[q].mH@c[p])
    for u,d in ((0,2),(1,3)): H=H+5*(n[u]@n[d]-.5*(n[u]+n[d]))
    E,V=torch.linalg.eigh(hermitian(H)); p=torch.softmax(-10*E,0)
    ce=V.mH@c[0]@V; weight=p[:,None]*ce.abs().square()
    times=torch.arange(41,dtype=RDTYPE)*.05
    f=(weight[:,:,None]*torch.exp(1j*(E[:,None]-E[None,:])[:,:,None]*times)).sum((0,1))
    return f,float((p*torch.diag(V.mH@n[0]@V).real).sum())


def from_lags(f):
    n=len(f);q=torch.arange(n)
    d=q[None,:]-q[:,None]
    return torch.where(d>=0,f[d.abs()],f[d.abs()].conj())


def write_csv(path,rows):
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def ladder(bits: int, orbital: int, create: bool) -> tuple[int, int] | None:
    if bool((bits >> orbital) & 1) == create:
        return None
    sign = -1 if (bits & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return bits ^ (1 << orbital), sign


def basis(L: int, nup: int, ndown: int) -> list[int]:
    return [sum(1 << j for j in u) + sum(1 << (L + j) for j in d)
            for u in combinations(range(L), nup)
            for d in combinations(range(L), ndown)]


def hamiltonian(L: int, nup: int, ndown: int, U: float,
                device: torch.device) -> tuple[torch.Tensor, list[int]]:
    """Open chain, t=1, mu=U/2. Spin-up orbitals precede spin-down."""
    bas = basis(L, nup, ndown)
    pos = {b: j for j, b in enumerate(bas)}
    H = torch.zeros((len(bas), len(bas)), dtype=RDTYPE, device="cpu")
    for j, bits in enumerate(bas):
        H[j, j] = U * sum(((bits >> k) & 1) * ((bits >> (L+k)) & 1)
                          for k in range(L)) - U * (nup + ndown) / 2
        for spin in (0, L):
            for k in range(L-1):
                for p, q in ((spin+k, spin+k+1), (spin+k+1, spin+k)):
                    a = ladder(bits, q, False)
                    b = ladder(a[0], p, True) if a else None
                    if b:
                        H[pos[b[0]], j] -= a[1] * b[1]
    check(float(torch.max(torch.abs(H-H.T))) < 1e-12, "Hamiltonian Hermitik degil.")
    return H.to(device=device, dtype=CDTYPE), bas


def build_problem(L: int, U: float, device: torch.device) -> dict[str, Any]:
    n, orbital = L // 2, L // 2 - 1
    H, bas = hamiltonian(L, n, n, U, device)
    ev, vec = torch.linalg.eigh(H)
    E0, ground = float(ev[0]), vec[:, 0]
    branches = []
    for create in (True, False):
        Hb, bb = hamiltonian(L, n + (1 if create else -1), n, U, device)
        b = torch.zeros(len(bb), dtype=CDTYPE, device=device)
        ix = {bits: j for j, bits in enumerate(bb)}
        for j, bits in enumerate(bas):
            out = ladder(bits, orbital, create)
            if out is not None:
                b[ix[out[0]]] = out[1] * ground[j]
        K = Hb - E0 * torch.eye(len(bb), dtype=CDTYPE, device=device)
        if not create:
            K = -K
        branches.append({"name": "add" if create else "remove", "K": K,
                         "b": b, "basis": bb, "energy_sign": 1 if create else -1})
    mass = sum(float(torch.vdot(q["b"], q["b"]).real) for q in branches)
    check(abs(mass-1.0) < 1e-10, "Elektron ekleme/cikarma agirliklari 1 etmiyor.")
    return {"L": L, "U": U, "E0": E0, "physical_E0": E0 + U*L/2,
            "orbital": orbital, "source_mass": mass, "ground_dimension": len(bas),
            "branches": branches}


def make_gates(L: int, bas: list[int], device: torch.device) -> list[dict[str, Any]]:
    pos = {bits: j for j, bits in enumerate(bas)}
    gates: list[dict[str, Any]] = []
    for spin in (0, 1):
        for site in range(L-1):
            p, q = spin*L+site, spin*L+site+1
            aidx, bidx, signs = [], [], []
            for col, bits in enumerate(bas):
                a = ladder(bits, q, False)
                b = ladder(a[0], p, True) if a else None
                if b:
                    aidx.append(col); bidx.append(pos[b[0]]); signs.append(a[1]*b[1])
            gates.append({"kind": "hop", "label": ["hop", spin, site],
                          "a": torch.tensor(aidx, dtype=torch.long, device=device),
                          "b": torch.tensor(bidx, dtype=torch.long, device=device),
                          "sign": torch.tensor(signs, dtype=RDTYPE, device=device)})
    for orbital in range(2*L):
        mask = [j for j, bits in enumerate(bas) if (bits >> orbital) & 1]
        gates.append({"kind": "phase", "label": ["occupation_phase", orbital],
                      "mask": torch.tensor(mask, dtype=torch.long, device=device)})
    for site in range(L):
        mask = [j for j, bits in enumerate(bas) if ((bits >> site)&1) and ((bits >> (L+site))&1)]
        gates.append({"kind": "phase", "label": ["double_occupation_phase", site],
                      "mask": torch.tensor(mask, dtype=torch.long, device=device)})
    return gates


def product_groups(L: int, branch: dict[str, Any]) -> tuple:
    gates = make_gates(L, branch['basis'], branch['b'].device)
    even = [g for g in gates if g['kind'] == 'hop' and g['label'][2] % 2 == 0]
    odd = [g for g in gates if g['kind'] == 'hop' and g['label'][2] % 2 == 1]
    onsite = [g for g in gates if g['label'][0] == 'double_occupation_phase']
    return even, odd, onsite



def positive_extension(f:torch.Tensor, floor:float=1e-10)->dict:
    """Center of each feasible one-point extension disk; AR(p) continuation.

    Matrix convention: T[i,j]=f[j-i], f[-k]=conj(f[k]). For p=len(f)-1,
    R=T[:p,:p].T and R a=f[1:] yield f[k]=sum_j a[j-1]f[k-j].
    A tiny, logged diagonal mixture restores strict positivity after roundoff.
    """
    f=f.to(CDTYPE).clone(); norm=float(f[0].real)
    check(norm>0 and len(f)>=2,'Invalid extension moments.')
    T=from_lags(f); mineig=float(torch.linalg.eigvalsh(T).min())
    target=floor*norm
    mix=max(0.,(target-mineig)/(norm-mineig)) if mineig<target else 0.
    f[1:]*=(1-mix);f[0]=norm
    p=len(f)-1
    R=from_lags(f[:-1]).T.contiguous()
    a=torch.linalg.solve(R,f[1:])
    fitres=float((R@a-f[1:]).abs().max())
    innovation=norm-float(torch.vdot(f[1:],a).real)
    # Companion roots are a computational sanity check, not model selection.
    companion=torch.zeros((p,p),dtype=CDTYPE);companion[0]=a
    if p>1:companion[1:,:-1]=torch.eye(p-1,dtype=CDTYPE)
    radius=float(torch.linalg.eigvals(companion).abs().max())
    check(radius<1+2e-6,'Unstable extension: no silent root clipping.')
    check(innovation>=-1e-7*norm,'Negative innovation variance.')
    # Generating-function numerator: F(z)=P(z)/(1-sum a_j z^j).
    numerator=f[:p].clone()
    for k in range(1,p): numerator[k]-=torch.dot(a[:k],f[:k].flip(0))
    return {'f':f,'a':a,'numerator':numerator,'audit':{
        'order':p,'diagonal_mixture':mix,'minimum_input_eigenvalue':mineig,
        'linear_solve_residual':fitres,'innovation_variance':innovation,
        'maximum_companion_root_modulus':radius,
        'choice':'Schur-disk center; maximum-entropy AR continuation',
        'not_unique_physical_extension':True}}


def extension_series(ext:dict,length:int)->torch.Tensor:
    f=ext['f'];a=ext['a'];p=len(a)
    out=torch.zeros(length,dtype=CDTYPE);n=min(len(f),length);out[:n]=f[:n]
    for k in range(n,length): out[k]=torch.dot(a,out[k-p:k].flip(0))
    return out


def extension_spectrum(ext:dict,omega:torch.Tensor,dt:float,eta:float)->torch.Tensor:
    """Exact Fourier series with exp(-eta |k| dt), not a fitted width."""
    zz=torch.exp((-eta+1j*omega)*dt);p=len(ext['a'])
    powers=zz[None,:]**torch.arange(p+1,dtype=RDTYPE)[:,None]
    denominator=1-(ext['a'][:,None]*powers[1:]).sum(0)
    check(float(denominator.abs().min())>1e-13,'Extension evaluation near numerical pole.')
    F=(ext['numerator'][:,None]*powers[:p]).sum(0)/denominator
    result=dt/(2*math.pi)*(2*F.real-float(ext['f'][0].real))
    check(bool(torch.isfinite(result).all()) and float(result.min())>-1e-6,
          'Positive extension yielded invalid spectrum; do not clip.')
    return result


def finite_spectrum(f,omega,dt,eta):
    j=torch.arange(1,len(f),dtype=RDTYPE)
    return dt/(2*math.pi)*(float(f[0].real)+2*(
        f[1:,None]*torch.exp((-eta+1j*omega[None,:])*dt*j[:,None])).sum(0).real)


def periodic_poles(energy,weight,omega,dt,eta):
    """Periodization of Lorentzians: integral over one period = sum(weight)."""
    den=math.cosh(eta*dt)-torch.cos((omega[None,:]-energy[:,None])*dt)
    return dt/(2*math.pi)*math.sinh(eta*dt)*(weight[:,None]/den).sum(0)


def apply_gate_many(v,gate,angle):
    out=v.clone()
    if gate['kind']=='hop':
        a,b,s=gate['a'],gate['b'],gate['sign']
        if v.ndim==2:s=s[:,None]
        out[a]=math.cos(angle)*v[a]-1j*math.sin(angle)*s*v[b]
        out[b]=math.cos(angle)*v[b]-1j*math.sin(angle)*s*v[a]
    else:out[gate['mask']]*=complex(math.cos(angle),-math.sin(angle))
    return out


def unitary_product(L,U,branch,dt,step):
    nstep=round(dt/step);check(abs(nstep*step-dt)<1e-12,'Noninteger product step.')
    groups=product_groups(L,branch);sgn=branch['energy_sign']
    W=torch.eye(len(branch['b']),dtype=CDTYPE)
    even,odd,onsite=groups;h=sgn*step
    for _ in range(nstep):
        for gates,angle in ((even,-h/2),(odd,-h/2),(onsite,U*h),(odd,-h/2),(even,-h/2)):
            for gate in gates:W=apply_gate_many(W,gate,angle)
    err=float((W.mH@W-torch.eye(len(W),dtype=CDTYPE)).abs().max())
    check(err<1e-10,'Product is not unitary.')
    return W


def unitary_poles(W,b,dt):
    """Jointly diagonalize commuting sine/cosine parts; verify the result."""
    cosine=(W+W.mH)/2; sine=(W-W.mH)/(2j)
    _,V=torch.linalg.eigh(hermitian(cosine+math.sqrt(2)*sine))
    D=V.mH@W@V;diag=D.diag()
    defect=float((D-torch.diag(diag)).abs().max())
    check(defect<2e-8 and float((diag.abs()-1).abs().max())<1e-8,
          'Unitary spectral decomposition unresolved; no unverified fallback.')
    return -torch.angle(diag)/dt,(V.mH@b).abs().square(),defect


def reference_problem(L,U,dt,step,nlags,omega):
    """CLASSICAL simulator/evaluator only; exact information never fits models."""
    problem=build_problem(L,U,torch.device('cpu'));result=[]
    for branch in problem['branches']:
        K,b=branch['K'],branch['b'];sgn=branch['energy_sign'];beta=float(torch.linalg.vector_norm(b))
        E,V=torch.linalg.eigh(K);weight=(V.mH@b).abs().square()
        W0=unitary_product(L,U,branch,dt,step)
        gamma=problem['E0']+U*(L+sgn)/2
        W=complex(math.cos(sgn*gamma*dt),math.sin(sgn*gamma*dt))*W0
        EW,wW,defect=unitary_poles(W,b,dt)
        f0=[];v=b/beta
        for _ in range(nlags+1):
            f0.append(torch.vdot(b/beta,v));v=W0@v
        f0=torch.stack(f0)
        phase=torch.exp(1j*sgn*gamma*dt*torch.arange(nlags+1,dtype=RDTYPE))
        analytic=((wW/beta**2)[:,None]*torch.exp(-1j*EW[:,None]*dt*
                   torch.arange(nlags+1,dtype=RDTYPE)[None,:])).sum(0)
        phase_error=float((phase*f0-analytic).abs().max());check(phase_error<2e-8,'Phase-reference error.')
        result.append({'f0':f0,'phase':phase,'beta2':beta**2,'E0':problem['E0'],
            'physical':periodic_poles(E,weight,omega,dt,ETA),
            'propagator':periodic_poles(EW,wW,omega,dt,ETA),
            'audit':{'phase_formula_max_error':phase_error,'unitary_diagonalization_defect':defect,
                'physical_weight_outside_principal_interval':float(weight[(E<-math.pi/dt)|(E>=math.pi/dt)].sum()),
                'gamma':gamma,'branch_sign':sgn}})
    return result


def reconstruct_time(f0,phase,omega,dt,eta):
    f=f0*phase
    clean,di=published_denoise(from_lags(f),list(range(len(f))),1.)
    ext=positive_extension(clean[0])
    return {'raw_time_window':finite_spectrum(f,omega,dt,eta),
        'psd_time_window':finite_spectrum(clean[0],omega,dt,eta),
        'psd_central_extension':extension_spectrum(ext,omega,dt,eta)},dict(denoising=di,extension=ext['audit'])


def sample_all_time(f,shots,seed,token):
    """NEW binomial simulation of S-only acquisition; private true mean."""
    g=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{token}|{seed}'.encode()).digest()[:8],'little')%(2**63-1))
    mu=torch.view_as_real(f[1:]).reshape(-1).clamp(-1,1)
    counts=torch.full(mu.shape,float(shots),dtype=RDTYPE)
    y=2*torch.binomial(counts,(1+mu)/2,generator=g)/shots-1
    return torch.cat((torch.ones(1,dtype=CDTYPE),torch.view_as_complex(y.reshape(-1,2).contiguous())))


def self_tests8b():
    tests={}
    w=torch.tensor([.3,.7],dtype=RDTYPE);E=torch.tensor([-.7,1.3],dtype=RDTYPE)
    dt=.25;eta=.2;times=torch.arange(16,dtype=RDTYPE)*dt
    f=(w[:,None]*torch.exp(-1j*E[:,None]*times)).sum(0)
    ext=positive_extension(f)
    extended=extension_series(ext,121)
    expected=(w[:,None]*torch.exp(-1j*E[:,None]*torch.arange(121,dtype=RDTYPE)*dt)).sum(0)
    tests['two_pole_extension_error']=float((extended-expected).abs().max())
    check(tests['two_pole_extension_error']<1e-6,'Exact two-pole extension failed.')
    omega=torch.linspace(-math.pi/dt,math.pi/dt,2401,dtype=RDTYPE)
    ar=extension_spectrum(ext,omega,dt,eta);pole=periodic_poles(E,w,omega,dt,eta)
    tests['analytic_fourier_max_error']=float((ar-pole).abs().max())
    check(tests['analytic_fourier_max_error']<2e-6,'Analytic Fourier mismatch.')
    tests['period_mass_error']=abs(float(integral(ar,omega))-1)
    check(tests['period_mass_error']<1e-7,'Periodized spectrum normalization failed.')
    # Independent Schur-complement feasible disk center.
    fp=torch.tensor([1.,.4+.2j,.1-.03j,.05+.02j],dtype=CDTYPE)
    ex=positive_extension(fp);middle=from_lags(ex['f'][:-1])
    disk=ex['f'][1:]@torch.linalg.solve(middle,ex['f'][1:].flip(0))
    predicted=torch.dot(ex['a'],ex['f'][1:].flip(0))
    tests['schur_disk_center_error']=float(abs(disk-predicted))
    check(tests['schur_disk_center_error']<1e-12,'Schur disk center mismatch.')
    long=extension_series(ex,60);eigmin=float(torch.linalg.eigvalsh(from_lags(long)).min())
    tests['extended_gram_minimum_eigenvalue']=eigmin;check(eigmin>-1e-9,'Extension not PSD.')
    # Explicitly demonstrate aliasing; no algorithm can resolve these from S alone.
    alias_E=E+2*math.pi/dt
    fa=(w[:,None]*torch.exp(-1j*alias_E[:,None]*times)).sum(0)
    tests['aliased_time_signal_difference']=float((f-fa).abs().max())
    check(tests['aliased_time_signal_difference']<1e-12,'Aliasing algebra failed.')
    # Published dimer normalization; the original figure noise is unavailable.
    fd,density=paper_dimer();tests['paper_dimer_G0']=float(fd[0].real);tests['paper_dimer_density']=density
    check(abs(density-.710556)<1e-6,'Published dimer normalization failed.')
    # Independent exact thermal spectrum from full Fock space, used only for evaluation.
    c=[annihilation(j) for j in range(4)];n=[a.mH@a for a in c]
    H=-2.3*sum(n)
    for p,q in ((0,1),(2,3)):H-=c[p].mH@c[q]+c[q].mH@c[p]
    for u,d in ((0,2),(1,3)):H+=5*(n[u]@n[d]-.5*(n[u]+n[d]))
    ev,V=torch.linalg.eigh(H);pop=torch.softmax(-10*ev,0);ce=V.mH@c[0]@V
    weights=(pop[:,None]*ce.abs().square()).flatten();energies=(ev[None,:]-ev[:,None]).flatten()
    td=torch.arange(201,dtype=RDTYPE)*.05
    exact_d=(weights[:,None]*torch.exp(-1j*energies[:,None]*td)).sum(0)
    dex=positive_extension(fd);dlong=extension_series(dex,201)
    tests['paper_dimer_exact_extension_t10_max_error']=float((dlong-exact_d).abs().max())
    check(tests['paper_dimer_exact_extension_t10_max_error']<2e-3,'Dimer extension failed.')
    dimer_rows=[];ww=torch.linspace(-math.pi/.05,math.pi/.05,12001,dtype=RDTYPE)
    target=periodic_poles(energies,weights,ww,.05,eta);norm=float(integral(target,ww))
    for seed in range(8):
        gen=torch.Generator().manual_seed(88100+seed)
        noise=.03*torch.randn((len(fd),2),generator=gen,dtype=RDTYPE)
        obs=fd+torch.view_as_complex(noise);obs[0]=fd[0]
        cleaned,di=published_denoise(from_lags(obs),list(range(len(obs))),float(fd[0].real))
        extended=positive_extension(cleaned[0]);s=extension_spectrum(extended,ww,.05,eta)
        r=finite_spectrum(obs,ww,.05,eta)
        dimer_rows.append({'seed':seed,'raw_spectral_L1':float(integral(abs(r-target),ww))/norm,
             'extended_spectral_L1':float(integral(abs(s-target),ww))/norm,
             'raw_time_mse':float((obs-fd).abs().square().mean()),
             'denoised_time_mse':float((cleaned[0]-fd).abs().square().mean())})
    tests['dimer_new_synthetic_noise']=dimer_rows
    tests['dimer_test_scope']='Authors thermal model; new synthetic noise, not figure-data replication.'
    tests['status']='PASS'
    return tests


def read_archived_f(raw):
    ct=torch.tensor(raw['counts_train'],dtype=RDTYPE);cv=torch.tensor(raw['counts_validation'],dtype=RDTYPE)
    yt=torch.tensor(raw['observed_train'],dtype=RDTYPE);yv=torch.tensor(raw['observed_validation'],dtype=RDTYPE)
    y=(ct*yt+cv*yv)/(ct+cv);ss={tuple(k):float(v) for k,v in zip(raw['keys'],y) if k[0]=='S'}
    lags=sorted({k[1] for k in ss});check(lags==list(range(1,max(lags)+1)),'Missing lag.')
    f=torch.tensor([1+0j]+[complex(ss[('S',d,'R')],ss[('S',d,'I')]) for d in lags],dtype=CDTYPE)
    total=int((ct+cv).sum());ns=2*len(lags)
    s_shots=int((ct+cv)[:ns].sum())
    return f,total,s_shots


def latent_periodic(snapshot,info,omega,dt):
    w=torch.tensor(snapshot['weights'],dtype=RDTYPE)
    A=torch.complex(torch.tensor(snapshot['energy_real'],dtype=RDTYPE),torch.tensor(snapshot['energy_imag'],dtype=RDTYPE))
    e,V=torch.linalg.eigh(hermitian(A));source=info['beta']*w.sqrt().to(CDTYPE)
    weights=(V.mH@source).abs().square();energies=info['sign']*(info['scale']*e-info['shift'])
    return periodic_poles(energies,weights,omega,dt,ETA)


def execute(args):
    import fcntl
    torch.set_num_threads(args.threads)
    root=args.out.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def log(s):
        print(s,flush=True)
        with (root/'run.log').open('a',encoding='utf8') as f:f.write(s+'\n')
    inp=args.input.expanduser().resolve();check(inp.is_file(),'Deney6 archive missing.')
    sig=hashlib.sha256(inp.read_bytes()).hexdigest()
    input7=args.report7.expanduser().resolve() if args.report7 is not None else None
    prior7={};sig7=None
    if input7 is not None:
        check(input7.is_file(),'Deney7 archive missing.')
        sig7=hashlib.sha256(input7.read_bytes()).hexdigest()
        with zipfile.ZipFile(input7) as z7:
            check(z7.testzip() is None,'Bad report7 CRC.')
            old_protocol=json.loads(z7.read('protocol.json'))
            check(old_protocol['input_sha256']==sig,'Experiment7 used a different input archive.')
            for name in z7.namelist():
                if name.startswith('checkpoints/') and name.endswith('.json'):
                    prior7[Path(name).stem]=json.loads(z7.read(name))
    cpdir=root/'checkpoints';cpdir.mkdir(exist_ok=True)
    (root/'script_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump_json(root/'environment.json',{'python':sys.version,'torch':str(torch.__version__),
        'device':'cpu','platform':platform.platform(),'threads':args.threads})
    log('[1] PSD uzatma, Fourier, aliasing ve yayin dimeri testleri...')
    tests=self_tests8b();dump_json(root/'self_tests.json',tests);log('Dogrulama: PASS')
    if args.suite=='test':return
    with zipfile.ZipFile(inp) as archive:
        check(archive.testzip() is None,'Bad input CRC.')
        cfg=json.loads(archive.read('config.json'));details=json.loads(archive.read('details.json'))
        cases=[(U,B,s) for U in cfg['U'] for B in cfg['budgets'] for s in cfg['seeds']]
        if args.suite=='sentinel':cases=[(U,B,cfg['seeds'][0]) for U in cfg['U'] for B in cfg['budgets']]
        if args.suite=='smoke':cases=[(5.,10**8,7000)]
        protocol={'version':VERSION,'input_sha256':sig,'report7_sha256':sig7,'suite':args.suite,'cases':cases,
            'spacing':SPACING,'eta':ETA,'metric':'L1 over ONE frequency period; not old nonperiodic metric',
            'physical_vs_propagator_targets_separate':True,'all_S_track':'new binomial simulation, NOT archived data',
            'source':SOURCE_URL,'extension_rule':'Schur disk center, diagonal mixture floor 1e-10',
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limits':['Exploratory, previously inspected model cases.','No same-information or equal-native-gate claim.',
            'Time-only reconstruction cannot resolve energy aliases without added assumptions.',
            'Original paper figure data/extension choice not reproduced.','No fits to exact reference.']}
        ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        if (root/'protocol.json').exists():
            check(json.loads((root/'protocol.json').read_text())['protocol_sha256']==ph,'Protocol changed; new --out required.')
        dump_json(root/'protocol.json',dict(protocol,protocol_sha256=ph))
        omega=torch.linspace(-math.pi/SPACING,math.pi/SPACING,2401,dtype=RDTYPE)
        refcache={};rows=[];clock=time.perf_counter()
        for ci,(U,B,seed) in enumerate(cases,1):
            key=f'U{U:g}_B{B}_seed{seed}';cp=cpdir/(key+'.json')
            if cp.exists():
                saved=json.loads(cp.read_text());check(saved['protocol_sha256']==ph,'Bad checkpoint.')
                rows.extend(saved['rows']);log(f'[{ci}/{len(cases)}] {key}: tamam, atlandi.');continue
            if U not in refcache:
                log(f'[2] U/t={U:g}: ideal H ve urun-unitary hedefleri ayri hesaplaniyor.')
                refcache[U]=reference_problem(6,U,SPACING,cfg['product_step'],23,omega)
                dump_json(root/f'reference_audit_U{U:g}.json',{'branches':[r['audit'] for r in refcache[U]]})
            refs=refcache[U];spectra={};audits=[];count_archive=0;count_s=0;count_allS=0;proxy_archive=0;proxy_allS=0
            for bid,ref in enumerate(refs):
                raw=json.loads(archive.read(f'raw_{key}_branch{bid}.json'))
                check(abs(raw['info']['beta']**2-ref['beta2'])<1e-8,'Source norm changed.')
                check(abs(raw['info']['shift']-(ref['E0']+U*6/4))<1e-8,'E0 mismatch.')
                f,total,used_s=read_archived_f(raw);count_archive+=total;count_s+=used_s
                ticks=[k[1] if k[0]=='S' else abs(raw['q'][k[1]])+abs(raw['q'][k[2]]) for k in raw['keys']]
                ncounts=[a+b for a,b in zip(raw['counts_train'],raw['counts_validation'])]
                rot_per_tick=round(SPACING/cfg['product_step'])*(4*(cfg['L']-1)+cfg['L'])
                proxy_archive+=sum(int(n)*int(t) for n,t in zip(ncounts,ticks))*rot_per_tick
                values,audit=reconstruct_time(f,ref['phase'],omega,SPACING,ETA)
                for name,valuespec in values.items():spectra[name+'_archived']=spectra.get(name+'_archived',0)+ref['beta2']*valuespec
                per=B//(2*46);fnew=sample_all_time(ref['f0'],per,seed,f'{U}|{B}|{bid}|8b_allS')
                values2,audit2=reconstruct_time(fnew,ref['phase'],omega,SPACING,ETA)
                for name,valuespec in values2.items():spectra[name+'_allS']=spectra.get(name+'_allS',0)+ref['beta2']*valuespec
                count_allS+=46*per;proxy_allS+=per*2*sum(range(1,24))*rot_per_tick
                old=details[key]['latent_diagonal_cv'][bid]['model']
                sp=latent_periodic(old,raw['info'],omega,SPACING)
                spectra['latent_diagonal_archived6_periodized']=spectra.get('latent_diagonal_archived6_periodized',0)+sp
                if input7 is not None:
                    cp7=prior7[key+f'_branch{bid}']['models']['diagonal']['selected']['model']
                    sp7=latent_periodic(cp7,raw['info'],omega,SPACING)
                    spectra['latent_diagonal_extended7_periodized']=spectra.get('latent_diagonal_extended7_periodized',0)+sp7
                audits.append({'archived':audit,'allS':audit2})
                dump_json(root/f'new_S_{key}_branch{bid}.json',{'f_real':fnew.real.tolist(),'f_imag':fnew.imag.tolist(),
                    'shots_per_component':per,'seed':seed,'scope':'new classical binomial simulation for same-budget S-only protocol'})
            physical=sum(r['physical'] for r in refs);prop=sum(r['propagator'] for r in refs)
            norm=float(integral(physical,omega));check(abs(norm-1)<1e-7,'Physical one-period mass incorrect.')
            check(count_archive<=B and count_allS<=B,'Shot budget exceeded.')
            rr=[]
            for method,spec in spectra.items():
                rr.append({'U':U,'budget':B,'seed':seed,'method':method,
                    'periodic_L1_physical':float(integral(abs(spec-physical),omega))/norm,
                    'periodic_L1_product_unitary':float(integral(abs(spec-prop),omega))/float(integral(prop,omega)),
                    'integral_mass':float(integral(spec,omega)),
                    'negative_mass':float(integral((-spec).clamp_min(0),omega)),
                    'total_charged_shots':count_allS if method.endswith('_allS') else count_archive,
                    'S_shots_used':count_allS if method.endswith('_allS') else count_s,
                    'new_classical_measurements':method.endswith('_allS'),
                    'shot_weighted_rotation_proxy':proxy_allS if method.endswith('_allS') else proxy_archive,
                    'periodic_H_vs_W_floor':float(integral(abs(physical-prop),omega))/norm})
            saved={'protocol_sha256':ph,'rows':rr,'branch_audits':audits}
            dump_json(cp,saved);rows.extend(rr);write_csv(root/'metrics.csv',rows)
            by={r['method']:r['periodic_L1_physical'] for r in rr}
            log(f'[{ci}/{len(cases)}] {key}: zaman-uzatma=%{100*by["psd_central_extension_archived"]:.3f}; '
                f'tum-butce-S=%{100*by["psd_central_extension_allS"]:.3f}; '
                f'mavi-periyodik=%{100*by["latent_diagonal_archived6_periodized"]:.3f}')
            if seed==cfg['seeds'][0]:
                write_csv(root/f'curves_{key}.csv',[{'omega':float(omega[k]),'physical_periodic':float(physical[k]),
                    'product_unitary_periodic':float(prop[k]),**{name:float(s[k]) for name,s in spectra.items()}} for k in range(len(omega))])
        summaries=[]
        for U,B in sorted({(r['U'],r['budget']) for r in rows}):
            for method in sorted({r['method'] for r in rows}):
                group=[r for r in rows if r['U']==U and r['budget']==B and r['method']==method]
                summaries.append({'U':U,'budget':B,'method':method,'n':len(group),
                    'median_periodic_L1_physical':statistics.median(r['periodic_L1_physical'] for r in group),
                    'median_periodic_L1_product':statistics.median(r['periodic_L1_product_unitary'] for r in group),
                    'median_negative_mass':statistics.median(r['negative_mass'] for r in group),
                    'physical_vs_unitary_L1':group[0]['periodic_H_vs_W_floor']})
        write_csv(root/'method_summary.csv',summaries)
        dump_json(root/'summary.json',{'status':'PASS','conditions_completed':len(cases),'rows':len(rows),
            'protocol_sha256':ph,'seconds_this_session':time.perf_counter()-clock,
            'scope':'Complete selected denoise-extension-Fourier route; NOT final fair physical-energy superiority benchmark.',
            'summary':summaries})
        log(f'SONUC: PASS — {len(cases)}/{len(cases)}; hedefler ve veri erisimi farkli, ustunluk iddiasi DEGIL.')
    dest=root.parent/(root.name+'_rapor.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name!='.lock':z.write(p,p.relative_to(root))
    log('RAPOR_ZIP: '+str(dest))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--report7',type=Path,default=None);p.add_argument('--out',type=Path,default=Path.home()/'spektrum_rakip8b')
    p.add_argument('--suite',choices=['test','smoke','sentinel','full'],default='full')
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.threads<=32:p.error('threads must be between 1 and 32')
    try:
        with torch.no_grad():execute(args)
    except (Exception,KeyboardInterrupt) as exc:
        print('HATA: '+str(exc),file=sys.stderr);traceback.print_exc();return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
