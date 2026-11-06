"""Matrix-free open Hubbard chain reference for experiment 9.
Only torch + standard library. No CPU complex dot/vdot, no dense full-sector H.
Ground/reference convergence checks do not constitute quantum advantage.
"""
from __future__ import annotations
import math,itertools
import torch
RD=torch.float64; CD=torch.complex128

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def spin_basis(L,n):
    return [sum(1<<j for j in c) for c in itertools.combinations(range(L),n)]

class Sector:
    def __init__(self,L,nu,nd,U,device):
        self.L=L;self.U=U;self.nu=nu;self.nd=nd;self.device=device
        self.up=spin_basis(L,nu);self.down=spin_basis(L,nd)
        self.basis=[a+(b<<L) for a in self.up for b in self.down]
        self.a=len(self.up);self.b=len(self.down);self.dimension=self.a*self.b
        def spinH(bs):
            ix={x:i for i,x in enumerate(bs)};H=torch.zeros((len(bs),len(bs)),dtype=RD)
            for j,x in enumerate(bs):
                for k in range(L-1):
                    if ((x>>k)&1)!=((x>>(k+1))&1):H[ix[x^(1<<k)^(1<<(k+1))],j]=-1.
            return H.to(device)
        self.hu=spinH(self.up);self.hd=spinH(self.down)
        self.diag=torch.tensor([[U*(a&b).bit_count()-U*(nu+nd)/2 for b in self.down] for a in self.up],dtype=RD,device=device)
        self.huc=self.hu.to(CD);self.hdc=self.hd.to(CD)
    def __matmul__(self,x):return self.apply(x)
    def apply(self,x):
        single=x.ndim==1
        require(x.shape[0]==self.dimension,'Sector dimension mismatch')
        v=x.reshape(self.a,self.b,-1)
        hu,hd=(self.huc,self.hdc) if x.is_complex() else (self.hu,self.hd)
        y=torch.einsum('ac,cbk->abk',hu,v)+torch.einsum('ack,bc->abk',v,hd)+self.diag[:,:,None]*v
        return y.reshape(self.dimension) if single else y.reshape(self.dimension,-1)

class Shifted:
    def __init__(self,H,E0,sign):self.H=H;self.E0=E0;self.sign=sign;self.dimension=H.dimension
    def __matmul__(self,x):return self.sign*(self.H@x-self.E0*x)

def tri(alpha,beta):
    a=torch.tensor(alpha,dtype=RD)
    T=torch.diag(a)
    if len(a)>1:
        b=torch.tensor(beta[:len(a)-1],dtype=RD);T+=torch.diag(b,1)+torch.diag(b,-1)
    return T

def lanczos_steps(H,v,maxiter=384):
    """Real fully reorthogonalized Lanczos; yields checkpoints and live basis."""
    require(not v.is_complex(),'Reference Lanczos expects a real source')
    D=v.numel();maxiter=min(maxiter,D)
    beta0=float(torch.linalg.vector_norm(v));require(beta0>0,'Zero source')
    Q=torch.zeros((D,maxiter),dtype=RD,device=v.device);Q[:,0]=v/beta0
    aa=[];bb=[]
    for k in range(maxiter):
        hq=H@Q[:,k];a=float((Q[:,k]*hq).sum());aa.append(a)
        w=hq-a*Q[:,k]
        if k:w=w-bb[k-1]*Q[:,k-1]
        for _ in range(2):w-=Q[:,:k+1]@(Q[:,:k+1].T@w)
        b=float(torch.linalg.vector_norm(w));bb.append(b)
        m=k+1;closed=(b<2e-12 or m==D)
        if m>=24 and (m%24==0 or closed or m==maxiter) or closed:
            yield m,tri(aa,bb),Q[:,:m],b,beta0,closed
        if closed:break
        if m<maxiter:Q[:,m]=w/b

def ground(H,seed=91273,maxiter=384,tol=2e-10):
    gen=torch.Generator(device='cpu').manual_seed(seed)
    v=torch.randn(H.dimension,generator=gen,dtype=RD).to(H.device)
    for m,T,Q,b,_,closed in lanczos_steps(H,v,maxiter):
        e,V=torch.linalg.eigh(T);psi=Q@V[:,0].to(Q.device)
        residual=float(torch.linalg.vector_norm(H@psi-float(e[0])*psi))
        if residual<tol:
            return float(e[0]),psi,{'dimension':H.dimension,'lanczos_dimension':m,'residual_norm':residual,'seed':seed,'global_ground_certified':False}
    raise RuntimeError(f'Ground reference not converged: m={m}, residual={residual:g}')

def spectral_poles(E,weights,omega,eta=.2):
    return (weights[:,None]*eta/(math.pi*((omega[None,:]-E[:,None])**2+eta*eta))).sum(0)

def periodic_poles(E,weights,omega,dt=.5,eta=.2):
    return dt/(2*math.pi)*math.sinh(eta*dt)*(weights[:,None]/(math.cosh(eta*dt)-torch.cos((omega[None,:]-E[:,None])*dt))).sum(0)

def response(H,b,omega,eta=.2,maxiter=960,tol=2e-9):
    """Galerkin residual bound at requested frequencies (plus periodic convergence).
    The bound is conditional on the numerically prepared source/E0, not their
    correctness and not an experimentally measured certificate.
    """
    b=b.real.to(RD);previous=None;wp=torch.linspace(-math.pi/.5,math.pi/.5,2401,dtype=RD)
    for m,T,Q,beta,beta0,closed in lanczos_steps(H,b,maxiter):
        e,V=torch.linalg.eigh(T);w=beta0**2*V[0,:]**2
        den=omega[None,:]+1j*eta-e[:,None]
        last=((V[-1,:]*V[0,:])[:,None]/den).sum(0)
        bound=float((beta0**2*beta*last.abs()/eta).max())
        periodic=periodic_poles(e,w,wp)
        pchange=float((periodic-previous).abs().max()) if previous is not None else float('inf')
        previous=periodic
        if bound<tol and (pchange<tol or closed):
            orth=float((Q.T@Q-torch.eye(m,device=Q.device,dtype=RD)).abs().max())
            # Direct Arnoldi relation check, including any reorthogonalization corrections.
            HQ=H@Q;res=HQ-Q@T.to(Q.device);expected=res.clone();expected[:,-1]=0
            relation_error=float(torch.linalg.vector_norm(expected))
            require(orth<1e-9 and relation_error<2e-9,'Lanczos identity/orthogonality failed')
            # Extra backward error contributes to a conservative numerical allowance.
            allowance=beta0**2*relation_error/eta**2
            require(bound+allowance<1e-7,'Reference numerical error allowance too large')
            return e,w,{'dimension':H.dimension,'lanczos_dimension':m,'maximum_resolvent_residual_bound':bound,
                'roundoff_allowance':allowance,'periodic_change':0. if closed and not math.isfinite(pchange) else pchange,
                'orthogonality_error':orth,'relation_error':relation_error,'source_norm_squared':beta0**2}
    raise RuntimeError(f'Spectrum reference did not converge: m={m}, bound={bound:g}, periodic_change={pchange:g}')

def make_problem(L,U,device):
    require(L in (4,6,8),'Only audited L=4,6,8 are supported')
    H=Sector(L,L//2,L//2,U,device);E0,g,gi=ground(H)
    # A second start guards accidental missed ground-state overlap.
    E02,g2,gi2=ground(H,seed=93331)
    require(abs(E0-E02)<1e-9,'Independent ground starts disagree')
    branches=[];orb=L//2-1
    for sign in (1,-1):
        Hb=Sector(L,L//2+sign,L//2,U,device);ix={x:i for i,x in enumerate(Hb.basis)}
        source=torch.zeros(Hb.dimension,dtype=RD)
        gg=g.cpu()
        for j,bits in enumerate(H.basis):
            occupied=(bits>>orb)&1
            if occupied==(0 if sign==1 else 1):
                s=-1 if (bits&((1<<orb)-1)).bit_count()%2 else 1
                source[ix[bits^(1<<orb)]]=s*gg[j]
        source=source.to(device)
        branches.append({'b':source.to(CD),'basis':Hb.basis,'K':Shifted(Hb,E0,sign),
                         'energy_sign':sign,'name':'add' if sign==1 else 'remove'})
    mass=sum(float((br['b'].abs()**2).sum()) for br in branches);require(abs(mass-1)<1e-10,'Source mass not one')
    return {'L':L,'U':U,'E0':E0,'physical_E0':E0+U*L/2,'source_mass':mass,
        'branches':branches,'ground_audit':{'first':gi,'second':gi2,'energy_difference':abs(E0-E02)}}
