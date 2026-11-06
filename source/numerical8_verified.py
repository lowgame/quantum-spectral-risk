#!/usr/bin/env python3
"""Experiment 8 continuation with stable central-extension evaluation.

Python 3.10+, PyTorch and standard library only. No installation/network access.
Original project files, measurements, rank selection, eta and diagonal floor
remain unchanged. Module-local CPU complex vector products avoid the diagnosed
PyTorch dot path. No global torch monkeypatch or package change is performed. The positive central extension uses the SAME mathematical
estimator, evaluated through Schur recursions instead of a cancelling rational
polynomial. Reflection coefficients use Python Decimal (70 digits).

A separate evaluation manifest records the numerical revision, while original
training checkpoint signatures are retained. Any old completed evaluations are
backed up and recomputed; raw data and training files are never rewritten by
this wrapper. Do NOT claim GPU/Orin validation from local CPU tests.
"""
from decimal import Decimal, localcontext
import math
import torch
CDTYPE=torch.complex128
RDTYPE=torch.float64

def _add(a,b): return (a[0]+b[0], a[1]+b[1])
def _sub(a,b): return (a[0]-b[0], a[1]-b[1])
def _mul(a,b): return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def _conj(a): return (a[0],-a[1])
def _cdecimal(z): return (Decimal.from_float(float(z.real)),Decimal.from_float(float(z.imag)))
def _complex(a): return complex(float(a[0]),float(a[1]))

def _levinson_decimal(f,precision=70):
    with localcontext() as ctx:
        ctx.prec=precision
        data=[_cdecimal(complex(v)) for v in f.tolist()]
        e=data[0][0]; zero=(Decimal(0),Decimal(0)); a=[]; reflections=[]
        if e<=0: raise ValueError('Nonpositive source norm')
        for k in range(1,len(data)):
            acc=zero
            for j in range(1,k): acc=_add(acc,_mul(a[j-1],data[k-j]))
            num=_sub(data[k],acc);g=(num[0]/e,num[1]/e)
            g2=g[0]*g[0]+g[1]*g[1]
            if g2>=1:
                raise ValueError(f'Schur positivity failed at order {k}: |reflection|^2={g2}')
            a=[_sub(a[j],_mul(g,_conj(a[k-2-j]))) for j in range(k-1)]+[g]
            reflections.append(g)
            e*=1-g2
        return ([ _complex(v) for v in a],[_complex(v) for v in reflections],float(e))

def stable_positive_extension(f,floor=1e-10):
    f=f.detach().to(device='cpu',dtype=CDTYPE).clone()
    norm=float(f[0].real)
    if not (len(f)>=2 and math.isfinite(norm) and norm>0 and bool(torch.isfinite(f).all())):
        raise ValueError('Invalid extension moments')
    if abs(float(f[0].imag))>1e-10*norm: raise ValueError('Complex equal-time norm')
    n=len(f);q=torch.arange(n);dd=q[None,:]-q[:,None]
    def gram(v): return torch.where(dd>=0,v[dd.abs()],v[dd.abs()].conj())
    mineig=float(torch.linalg.eigvalsh(gram(f)).min())
    # Unchanged diagonal loading rule from original implementation.
    target=floor*norm
    mix=max(0.,(target-mineig)/(norm-mineig)) if mineig<target else 0.
    f[1:]*=(1-mix);f[0]=norm
    aa,gg,innovation=_levinson_decimal(f)
    a=torch.tensor(aa,dtype=CDTYPE);g=torch.tensor(gg,dtype=CDTYPE)
    p=len(a);R=gram(f)[:p,:p].T.contiguous()
    residual=float((R@a-f[1:]).abs().max())
    companion=torch.zeros((p,p),dtype=CDTYPE);companion[0]=a
    if p>1: companion[1:,:-1]=torch.eye(p-1,dtype=CDTYPE)
    radius=float(torch.linalg.eigvals(companion).abs().max())
    numerator=f[:p].clone()
    for k in range(1,p):
        av=[complex(v) for v in a[:k].tolist()]
        fv=[complex(v) for v in f[:k].flip(0).tolist()]
        numerator[k]-=python_complex_sum(x*y for x,y in zip(av,fv))
    return {'f':f,'a':a,'numerator':numerator,'reflection':g,'audit':{
        'order':p,'diagonal_mixture':mix,'minimum_input_eigenvalue':mineig,
        'linear_solve_residual':residual,'innovation_variance':innovation,
        'maximum_companion_root_modulus':radius,
        'maximum_reflection_modulus':float(g.abs().max()),
        'coefficient_precision_decimal_digits':70,
        'choice':'Schur-disk center; maximum-entropy AR continuation',
        'not_unique_physical_extension':True,
        'numerical_revision':'Levinson 70 digits + Schur + no native CPU complex dot',
        'spectrum_or_root_clipping':False}}

def stable_extension_spectrum(ext,omega,dt,eta):
    if not (math.isfinite(dt) and math.isfinite(eta) and dt>0 and eta>0):
        raise ValueError('Need positive dt and eta')
    if 'reflection' not in ext: raise ValueError('Missing stable reflection coefficients')
    omega=omega.to(dtype=RDTYPE)
    z=torch.exp((-eta+1j*omega)*dt)
    s=torch.zeros_like(z)
    for g in reversed(ext['reflection'].tolist()):
        zs=z*s
        den=1+g.conjugate()*zs
        if float(den.abs().min())<1e-14: raise ValueError('Schur recursion near numerical pole')
        s=(g+zs)/den
    u=z*s
    result=dt/(2*math.pi)*float(ext['f'][0].real)*(1-u.abs().square())/(1-u).abs().square()
    if not bool(torch.isfinite(result).all()) or float(result.min())<0:
        raise ValueError(f'Stable positive extension failed; min={float(result.min())}; no clipping applied')
    return result


import argparse
import fcntl
import hashlib
import importlib.util
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REVISION = '8c-schur-cpu-dot-2'
EXPECTED = {
    'core6.py': '519d65fd8774f49dc562e6fb733302ab96a8082fe52071edfc07ff125d483306',
    'spektrum_mekanizma8.py': 'a76d41366191d0464e52338d067c522353136476c4f3d210971dc05e35f1bd73',
    'spektrum_rakip8b.py': '05f9e41d768c2d4b337801a2f6e8e8079d09d7e42841fccd6b3ef9f2a745f465',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(path)

def require(ok,message):
    if not ok: raise RuntimeError(message)

def tree_digests(root, subdirs=('raw','fits')):
    return {str(p.relative_to(root)):sha(p) for sub in subdirs
            for p in sorted((root/sub).rglob('*.json')) if p.is_file()}

def load_driver(project):
    for filename,expected in EXPECTED.items():
        path=project/filename
        require(path.is_file(),f'Dosya yok: {path}')
        require(sha(path)==expected,
                f'Beklenmeyen kod surumu: {filename}. Eski dosyayi degistirmeyin; hata ciktisini paylasin.')
    sys.path.insert(0,str(project))
    spec=importlib.util.spec_from_file_location('mechanism8_original',project/'spektrum_mekanizma8.py')
    require(spec is not None and spec.loader is not None,'Deney modulu acilamadi.')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    return driver

def validate_old_protocol(root,args):
    path=root/'protocol.json'
    require(path.exists(),'Mevcut deneyin protocol.json dosyasi yok; bu komut bir devam duzeltmesidir.')
    protocol=json.loads(path.read_text())
    body={k:v for k,v in protocol.items() if k!='protocol_sha256'}
    calculated=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    require(calculated==protocol['protocol_sha256'],'Kayitli protokol ozeti bozuk.')
    for field,name in [('code_sha256','spektrum_mekanizma8.py'),('core_sha256','core6.py'),
                       ('rival_sha256','spektrum_rakip8b.py')]:
        require(protocol[field]==EXPECTED[name],f'Protokolde beklenmeyen {field}.')
    require(protocol['suite']==args.suite,'--suite mevcut deneyle ayni olmali.')
    return protocol

def prepare_revision(root,protocol,self_hash):
    """Idempotent, recoverable backup of evaluations. Never alter training/raw."""
    manifest_path=root/'evaluation_revision.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['revision'] != REVISION:
            require(manifest['revision']=='8c-schur-numerics-1' and manifest['status']=='active',
                    'Bilinmeyen veya yarim kalmis eski duzeltme kaydi; dosyalara dokunulmadi.')
            for rel,digest in manifest.get('preserved_input_sha256',{}).items():
                require((root/rel).is_file() and sha(root/rel)==digest,'Old revision preservation mismatch.')
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            archive=root/'numerical_audits'/('previous_evaluation_revision_'+stamp+'.json')
            archive.parent.mkdir(exist_ok=True)
            shutil.copy2(manifest_path,archive)
            manifest_path.unlink()
            return prepare_revision(root,protocol,self_hash)
        require(manifest['runner_sha256']==self_hash,'Farkli v2 dosyasi: mevcut sonuc uzerine yazilmadi.')
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=root.parent/(root.name+'_before_'+REVISION+'_'+stamp)
        backup.mkdir(parents=True,exist_ok=False)
        shutil.copy2(root/'protocol.json',backup/'protocol.json')
        originals=tree_digests(root)
        immutable={rel:digest for rel,digest in originals.items() if rel.startswith('raw/')}
        partial={}
        for rel,digest in originals.items():
            if rel.startswith('fits/'):
                records=json.loads((root/rel).read_text())['records']
                done={r['nominal_iterations'] for r in records}
                if set(protocol['milestones']).issubset(done):immutable[rel]=digest
                else:partial[rel]=digest
        save_json(backup/'raw_and_fit_sha256.json',originals)
        for name in ('run.log','environment.json','self_tests.json'):
            if (root/name).is_file():shutil.copy2(root/name,backup/name)
        manifest={'revision':REVISION,'runner_sha256':self_hash,'status':'migrating_evaluations',
            'training_protocol_sha256':protocol['protocol_sha256'],
            'original_code_sha256':EXPECTED,'backup_directory':str(backup),
            'preserved_input_sha256':immutable,'incomplete_fits_allowed_to_resume':partial,
            'scope':'Central extension plus module-local CPU scalar vector arithmetic; no rank/optimizer/data changes; saved models audited.',
            'diagonal_floor':1e-10,'decimal_precision':70,'spectrum_clipping':False,
            'new_measurements':False,'eta_changed':False,'state_preparation_changed':False,
            'started_utc':stamp}
        save_json(manifest_path,manifest)
    if manifest['status']=='migrating_evaluations':
        backup=Path(manifest['backup_directory'])
        for name in ('conditions','curves','metrics.csv','summary.json'):
            p=root/name;dest=backup/name
            if p.exists():
                require(not dest.exists(),f'Yedekleme cakismasi: {dest}')
                shutil.move(str(p),str(dest))
        for rel,digest in manifest['preserved_input_sha256'].items():
            require((root/rel).is_file() and sha(root/rel)==digest,
                    f'Korunacak egitim/ham dosya degismis: {rel}')
        manifest['status']='active'
        save_json(manifest_path,manifest)
    return manifest

# Only this process's project-module references are adapted; installed PyTorch
# and original project files are not edited.
import ast
import cmath
import inspect
import textwrap


def safe_vector_product(a, b, conjugate=False, out=None):
    """CPU complex dot through real arithmetic; preserves dot/vdot distinction.

    No complex BLAS dot is called. The expression remains differentiable.
    Non-CPU/non-complex operations retain the ordinary PyTorch implementation.
    """
    require(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), 'Tensor inputs required.')
    require(a.ndim == b.ndim == 1 and a.numel() == b.numel(), 'Dot inputs must have equal 1D shapes.')
    require(a.dtype == b.dtype and a.device == b.device, 'Dot dtype/device mismatch.')
    if a.device.type != 'cpu' or not a.is_complex():
        fn = torch.vdot if conjugate else torch.dot
        return fn(a, b) if out is None else fn(a, b, out=out)
    # .real/.imag respect PyTorch's lazy conjugate and negative views.
    ar, ai, br, bi = a.real, a.imag, b.real, b.imag
    if conjugate:
        real = (ar * br + ai * bi).sum()
        imag = (ar * bi - ai * br).sum()
    else:
        real = (ar * br - ai * bi).sum()
        imag = (ar * bi + ai * br).sum()
    result = torch.complex(real, imag)
    if out is not None:
        require(not a.requires_grad and not b.requires_grad, 'out + autograd is not supported by the local adapter.')
        require(out.dtype == result.dtype and out.device == result.device, 'Dot out mismatch.')
        out.copy_(result)
        return out
    return result


class ProjectTorch:
    """Module-local facade; never changes torch.dot globally."""
    def __getattr__(self, name):
        return getattr(torch, name)

    def dot(self, input, tensor, *, out=None):
        return safe_vector_product(input, tensor, False, out)

    def vdot(self, input, other, *, out=None):
        return safe_vector_product(input, other, True, out)


def safe_matmul(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.ndim == b.ndim == 1 and a.device.type == b.device.type == 'cpu' and a.is_complex():
            return safe_vector_product(a, b)
    return a @ b


def python_complex_sum(values):
    values = list(values)
    return complex(math.fsum(v.real for v in values), math.fsum(v.imag for v in values))


def scalar_extension_series(ext, length):
    """Same AR recurrence, without the failing CPU complex-dot kernel."""
    require(isinstance(length, int) and length >= 0, 'Invalid series length.')
    coefficients = [complex(v) for v in ext['a'].detach().cpu().tolist()]
    moments = [complex(v) for v in ext['f'].detach().cpu().tolist()]
    require(len(moments) == len(coefficients)+1, 'Invalid recurrence order.')
    values = moments[:length]
    p = len(coefficients)
    for k in range(len(values), length):
        value = python_complex_sum(coefficients[j]*values[k-1-j] for j in range(p))
        require(math.isfinite(value.real) and math.isfinite(value.imag), 'Nonfinite recurrence; no clipping.')
        values.append(value)
    return torch.tensor(values, dtype=CDTYPE)


def install_local_adapters(driver):
    """Leave test inputs/assertions/tolerances intact; replace vector arithmetic.

    The original self-test also uses vector @ vector, which can dispatch to the
    same complex-dot kernel. Its AST is rewritten ONLY for MatMult operators.
    The helper leaves matrix products unchanged. All other statements and all
    comparisons are byte-for-byte equivalent at AST level.
    """
    rival, core = driver.rival, driver.core
    original = rival.self_tests8b
    source = textwrap.dedent(inspect.getsource(original))
    original_tree = ast.parse(source)
    before_asserts = [ast.dump(node, include_attributes=False)
                      for node in ast.walk(original_tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'check']
    class VectorDispatch(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.MatMult):
                return ast.copy_location(ast.Call(func=ast.Name(id='_safe_matmul8', ctx=ast.Load()),
                                                   args=[node.left, node.right], keywords=[]), node)
            return node
    tree = ast.fix_missing_locations(VectorDispatch().visit(ast.parse(source)))
    after_asserts = [ast.dump(node, include_attributes=False)
                     for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'check']
    require(before_asserts == after_asserts, 'Self-test assertions changed; refusing patch.')
    facade = ProjectTorch()
    rival.torch = facade
    core.torch = facade
    rival.positive_extension = stable_positive_extension
    rival.extension_series = scalar_extension_series
    rival.extension_spectrum = stable_extension_spectrum
    namespace = dict(rival.__dict__)
    namespace['_safe_matmul8'] = safe_matmul
    exec(compile(tree, '<hash-checked-rival-selftest-vector-adapter>', 'exec'), namespace)
    rival.self_tests8b = namespace['self_tests8b']
    return {'scope':'module-local CPU complex dot/vdot; vector @ only in rival self-test',
            'global_torch_modified':False, 'project_files_modified':False,
            'selftest_original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'selftest_adapted_ast_sha256':hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
            'test_assertions_unchanged':True, 'assertion_count':len(before_asserts)}


def arithmetic_audit(device='cpu'):
    """Native dot anomalies are recorded. All OTHER tested paths must pass."""
    dev = torch.device(device)
    records=[]
    for n in (1,2,4,7,15,16,23,24,40,64):
        aa=[complex(math.sin(i+.2),math.cos(2*i+.3)) for i in range(n)]
        bb=[complex(math.cos(.7*i),math.sin(.4*i+.1)) for i in range(n)]
        a=torch.tensor(aa,dtype=CDTYPE,device=dev);b=torch.tensor(bb,dtype=CDTYPE,device=dev)
        expect=python_complex_sum(x*y for x,y in zip(aa,bb))
        ev=python_complex_sum(x.conjugate()*y for x,y in zip(aa,bb))
        row={'n':n}
        for name, fn, value in [('native_dot',lambda:torch.dot(a,b),expect),
                                ('native_vdot',lambda:torch.vdot(a,b),ev),
                                ('native_vector_matmul',lambda:a@b,expect)]:
            try:
                diff=abs(complex(fn())-value)
                row[name+'_error']=diff if math.isfinite(diff) else None
                if not math.isfinite(diff):row[name+'_exception']='Nonfinite result'
            except Exception as exc: row[name+'_exception']=str(exc)
        row['safe_dot_error']=abs(complex(safe_vector_product(a,b))-expect)
        row['safe_vdot_error']=abs(complex(safe_vector_product(a,b,True))-ev)
        row['elementwise_sum_error']=abs(complex((a*b).sum())-expect)
        for name in ('safe_dot_error','safe_vdot_error','elementwise_sum_error'):
            require(row[name] < 2e-11, f'{device}: arithmetic guard failed: {name}, n={n}, error={row[name]}')
        # Non-contiguous and lazy-conjugate views.
        aa2=torch.stack((a,a),dim=1)[:,0].conj()
        row['safe_conjugate_strided_error']=abs(complex(safe_vector_product(aa2,b))-ev)
        require(row['safe_conjugate_strided_error']<2e-11,'Conjugate/stride dot failed.')
        records.append(row)
    re=torch.tensor([.2,.3,-.7,1.5],dtype=RDTYPE,device=dev)
    rexpect=math.fsum(float(x)**2 for x in re.cpu().tolist())
    real_error=abs(float(torch.dot(re,re))-rexpect)
    require(real_error<1e-12,'Real dot is also incorrect; training stopped.')
    # Compare matrix/matrix, matrix/vector and vector/matrix against Python sums.
    aa=[[complex(math.sin(i+.3*j),math.cos(.2*i-j)) for j in range(7)] for i in range(6)]
    bb=[[complex(math.cos(i+.7*j),math.sin(.5*i+j)) for j in range(5)] for i in range(7)]
    A=torch.tensor(aa,dtype=CDTYPE,device=dev);B=torch.tensor(bb,dtype=CDTYPE,device=dev)
    target=torch.tensor([[python_complex_sum(aa[i][k]*bb[k][j] for k in range(7))
                         for j in range(5)] for i in range(6)],dtype=CDTYPE,device=dev)
    mm=float((A@B-target).abs().max())
    vv=torch.tensor([complex(.2*i,math.sin(i+.1)) for i in range(7)],dtype=CDTYPE,device=dev)
    mv_target=torch.tensor([python_complex_sum(aa[i][k]*complex(vv[k]) for k in range(7)) for i in range(6)],dtype=CDTYPE,device=dev)
    vm_target=torch.tensor([python_complex_sum(complex(vv[k])*bb[k][j] for k in range(7)) for j in range(5)],dtype=CDTYPE,device=dev)
    mv=float((A@vv-mv_target).abs().max());vm=float((vv@B-vm_target).abs().max())
    require(max(mm,mv,vm)<2e-11,f'{device}: matrix arithmetic also failed; no training. errors={mm,mv,vm}')
    # Hermitian eigenpairs and complex solve checked by residual + real block solve.
    H=A@A.mH+torch.eye(6,dtype=CDTYPE,device=dev)
    val,V=torch.linalg.eigh(H)
    eigerr=float((H@V-V*val[None,:]).abs().max())
    orth=float((V.mH@V-torch.eye(6,dtype=CDTYPE,device=dev)).abs().max())
    rhs=mv_target
    x=torch.linalg.solve(H,rhs)
    block=torch.cat((torch.cat((H.real,-H.imag),1),torch.cat((H.imag,H.real),1)),0)
    xr=torch.linalg.solve(block,torch.cat((rhs.real,rhs.imag)))
    solveerr=float((x-torch.complex(xr[:6],xr[6:])).abs().max())
    require(max(eigerr,orth,solveerr)<2e-10,f'{device}: linear-algebra guard failed.')
    # Complex matmul/backprop used in fitting: compare autograd to real finite differences.
    Ar=A.real.clone().requires_grad_(True);Ai=A.imag.clone().requires_grad_(True)
    def objective():return (torch.complex(Ar,Ai)@B).abs().square().sum()
    with torch.enable_grad():
        loss=objective();gr,gi=torch.autograd.grad(loss,(Ar,Ai))
    errors=[]
    for p,g,i,j in [(Ar,gr,2,3),(Ai,gi,4,5)]:
        with torch.no_grad():
            original=float(p[i,j]);eps=1e-5
            p[i,j]=original+eps;plus=float(objective())
            p[i,j]=original-eps;minus=float(objective())
            p[i,j]=original
        expected=(plus-minus)/(2*eps);actual=float(g[i,j])
        errors.append(abs(actual-expected)/max(1.,abs(actual),abs(expected)))
    require(max(errors)<5e-7,f'{device}: gradient guard failed.')
    return {'status':'PASS','device':str(dev),'dot_records':records,'native_dot_max_error':
             max((r.get('native_dot_error') or 0.) for r in records),
            'real_dot_error':real_error,'matrix_product_max_errors':{'mm':mm,'mv':mv,'vm':vm},
            'eigen_residual':eigerr,'eigen_orthogonality':orth,'complex_vs_real_block_solve_error':solveerr,
            'gradient_max_relative_error':max(errors),'not_exhaustive_environment_certification':True}


def independent_model_means(saved, raw):
    """Python complex/scalar reference; no Torch dot/matmul/linear algebra."""
    weights=saved['weights'];phase=saved['phases'];rank=len(weights)
    A=[[complex(r,i) for r,i in zip(rr,ii)] for rr,ii in zip(saved['energy_real'],saved['energy_imag'])]
    V=[[math.sqrt(weights[a])*cmath.exp(-1j*phase[a]*q) for q in raw['q']] for a in range(rank)]
    AV=[[python_complex_sum(A[a][b]*V[b][j] for b in range(rank)) for j in range(len(raw['q']))] for a in range(rank)]
    means=[]
    for key in raw['keys']:
        if key[0]=='S':v=python_complex_sum(weights[a]*cmath.exp(-1j*phase[a]*key[1]) for a in range(rank))
        else:
            i,j=key[1:3];v=python_complex_sum(V[a][i].conjugate()*AV[a][j] for a in range(rank))
        means.append(v.real if key[-1]=='R' else v.imag)
    return means


def python_deviance(pred, observations, counts):
    terms=[]
    for mu,obs,n in zip(pred,observations,counts):
        p=min(1-1e-12,max(1e-12,(1+mu)/2));q=min(1.,max(0.,(1+obs)/2))
        entropy=0.
        if q>0:entropy+=q*math.log(max(q,1e-300))
        if q<1:entropy+=(1-q)*math.log1p(-min(q,1-1e-15))
        terms.append(n*(entropy-q*math.log(p)-(1-q)*math.log1p(-p)))
    return math.fsum(terms)/sum(counts)


def check_saved_fits(driver, root, protocol):
    """Audit every saved model; no refitting and no changes to raw/checkpoints."""
    rows=[];rawcache={}
    for path in sorted((root/'fits').glob('*.json')):
        stored=json.loads(path.read_text())
        require(stored['protocol_sha256']==protocol['protocol_sha256'],f'Fit protocol mismatch: {path.name}')
        tag=path.stem.split('_diag_r')[0].split('_general_r')[0]
        rawpath=root/'raw'/(tag+'.json')
        if tag not in rawcache:rawcache[tag]=json.loads(rawpath.read_text())
        raw=rawcache[tag]
        rawhash=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
        require(stored['raw_sha256']==rawhash,'Fit/raw identity mismatch.')
        layout=driver.core.SettingLayout(raw['q'],torch.device('cpu'))
        for rec in stored['records']:
            mu=independent_model_means(rec['model'],raw)
            with torch.no_grad():
                model=driver.restore(layout,rec['model'],rec['diagonal']);computed=model().tolist()
            meanerr=max(abs(a-b) for a,b in zip(mu,computed))
            losses={stage:python_deviance(mu,raw['observed_'+rawstage],raw['counts_'+rawstage])
                    for stage,rawstage in [('training','train'),('selection','validation')]}
            lossdiff=max(abs(losses[stage]-rec[stage+'_deviance']) for stage in ('training','selection'))
            require(meanerr<2e-8 and lossdiff<1e-9,
                    f'Saved-model audit failed: {path.name}, step={rec["nominal_iterations"]}; records preserved, no training.')
            rows.append({'file':path.name,'nominal_iterations':rec['nominal_iterations'],
                         'python_vs_torch_forward_max_error':meanerr,'python_vs_recorded_loss_max_error':lossdiff})
    return {'status':'PASS','model_records':len(rows),'files':len(list((root/'fits').glob('*.json'))),
            'maximum_forward_difference':max((r['python_vs_torch_forward_max_error'] for r in rows),default=0.),
            'maximum_recorded_loss_difference':max((r['python_vs_recorded_loss_max_error'] for r in rows),default=0.),
            'records':rows,'scope':'forward/loss integrity; does not prove optimizer convergence or all past gradients'}


def run(args):
    project=args.project.expanduser().resolve();root=args.out.expanduser().resolve()
    require(root.is_dir(),f'Mevcut deney klasoru bulunamadi: {root}')
    torch.set_num_threads(args.threads)
    # Lock this wrapper; additionally check the original driver's lock before migration.
    wrapper_lock=(root/'.numerical_fix.lock').open('a')
    try:fcntl.flock(wrapper_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Duzeltmeli deney zaten calisiyor; ikinci kez baslatilmadi.')
    guard=(root/'.lock').open('a')
    try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Orijinal deney halen calisiyor; dosyalara dokunulmadi.')
    driver=load_driver(project);rival=driver.rival;core=driver.core
    protocol=validate_old_protocol(root,args)
    old_positive=rival.positive_extension;old_spectrum=rival.extension_spectrum
    print('[DUZELTME V2] CPU karmasik dot/vdot icin yerel guvenli hesap; test esikleri ayni.',flush=True)
    before=tree_digests(root)
    guard_report=arithmetic_audit('cpu')
    adapters=install_local_adapters(driver)
    print(f'[KONTROL] Yerel vektor hesabi, matris islemleri ve turev: PASS; native dot farki={guard_report["native_dot_max_error"]:.6g}',flush=True)
    with torch.no_grad():
        tests=rival.self_tests8b()
    print(f'[KONTROL] Iki-kutup hatasi={tests["two_pole_extension_error"]:.12g}; esik=1e-6: PASS.',flush=True)
    print('[KONTROL] Fourier, agirlik, Schur ve dimer testleri: PASS.',flush=True)
    fit_audit=check_saved_fits(driver,root,protocol)
    require(tree_digests(root)==before,'Preflight changed raw/fit files.')
    print(f'[KONTROL] Kayitli {fit_audit["files"]} egitim dosyasi / {fit_audit["model_records"]} model: bagimsiz kayip kontrolu PASS.',flush=True)
    # Run regression directly on the user's ALREADY SAVED data, before any training.
    checks=[];omega=torch.linspace(-math.pi/.5,math.pi/.5,2401,dtype=RDTYPE)
    for path in sorted((root/'raw').glob('*.json')):
        raw=json.loads(path.read_text())
        require(raw['protocol_sha256']==protocol['protocol_sha256'],f'Ham protokol farkli: {path.name}')
        f,_,_=rival.read_archived_f(raw);sign=raw['info']['sign']
        L,U=protocol['L'],protocol['U']
        E0=raw['info']['shift']-U*L/4
        gamma=E0+U*(L+sign)/2
        phase=torch.exp(1j*sign*gamma*.5*torch.arange(len(f),dtype=RDTYPE))
        clean,denoise=rival.published_denoise(rival.from_lags(f*phase),list(range(len(f))),1.)
        old_error=None;old_delta=None
        try:
            exold=old_positive(clean[0]);s_old=old_spectrum(exold,omega,.5,.2)
        except (RuntimeError,ValueError) as ex:
            old_error=str(ex)
        exnew=stable_positive_extension(clean[0]);snew=stable_extension_spectrum(exnew,omega,.5,.2)
        mass=float(rival.integral(snew,omega))
        require(abs(mass-1)<1e-7,f'Spektral agirlik testi gecmedi: {path.name}')
        if old_error is None:old_delta=float((s_old-snew).abs().max())
        row={'raw_file':path.name,'raw_sha256':sha(path),'old_failure':old_error,
             'old_new_max_difference_if_old_valid':old_delta,
             'new_minimum_spectrum':float(snew.min()),'new_period_mass':mass,
             'extension':exnew['audit'],'denoising':denoise}
        checks.append(row)
        if old_error:print(f'[KONTROL] {path.name}: eski hesap durdu; duzeltilmis hesap PASS.',flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    save_json(root/'numerical_audits'/f'preflight_{stamp}.json',
              {'status':'PASS','tests':tests,'saved_measurement_checks':checks,
               'torch':str(torch.__version__),'device':'cpu','revision':REVISION,
                'arithmetic':guard_report,'local_adapters':adapters,'saved_fit_audit':fit_audit})
    print(f'[KONTROL] Matematik testleri ve kayitli {len(checks)} ham dal: PASS.',flush=True)
    if args.check_only:
        print('[KONTROL] Tamam. --check-only: egitim/degerlendirme baslatilmadi.',flush=True)
        return 0
    self_hash=sha(__file__);manifest=prepare_revision(root,protocol,self_hash)
    if Path(__file__).resolve()!=(root/'spektrum_devam8_v2.py').resolve():
        shutil.copyfile(Path(__file__),root/'spektrum_devam8_v2.py')
    (root/'numerical_audits').mkdir(exist_ok=True)
    fits=list((root/'fits').glob('*.json'));complete=0
    for fp in fits:
        data=json.loads(fp.read_text())
        require(data['protocol_sha256']==protocol['protocol_sha256'],'Fit protokol ozeti farkli.')
        done={r['nominal_iterations'] for r in data['records']}
        if set(protocol['milestones']).issubset(done):complete+=1
    print(f'[DEVAM] {complete} tamamlanmis egitim dizisi korunuyor; bunlar tekrar egitilmeyecek.',flush=True)
    print('[DEVAM] Orijinal egitim kodu, ham olcumler ve eta degismedi. Eski degerlendirmeler yedeklendi.',flush=True)
    # Record numerical provenance in newly emitted summaries and condition files.
    original_dump=core.dump_json
    def audited_dump(path,value):
        p=Path(path)
        if p.name=='summary.json' or p.parent==root/'conditions':
            value=dict(value,evaluation_revision=REVISION,evaluation_runner_sha256=self_hash,
                       original_training_protocol_sha256=protocol['protocol_sha256'])
        return original_dump(path,value)
    core.dump_json=audited_dump
    original_reconstruct=rival.reconstruct_time
    def audited_reconstruct(f,phase,omega,dt,eta):
        values,audit=original_reconstruct(f,phase,omega,dt,eta)
        spec=values['psd_central_extension']
        is_period=abs(float(omega[-1]-omega[0])-2*math.pi/dt)<1e-8
        mass=float(rival.integral(spec,omega)) if is_period else None
        if is_period:require(abs(mass-float(f[0].real))<1e-7,'Merkezi uzatma toplam agirligi degisti.')
        moments=f*phase
        moment_data={'real':moments.real.tolist(),'imag':moments.imag.tolist()}
        token=hashlib.sha256(json.dumps(moment_data,sort_keys=True).encode()).hexdigest()
        save_json(root/'numerical_audits'/f'extension_{token[:24]}.json',
                  {'revision':REVISION,'dt':dt,'eta':eta,'moments':moment_data,
                   'minimum_spectrum':float(spec.min()),'period_mass':mass,'audit':audit})
        return values,audit
    rival.reconstruct_time=audited_reconstruct
    # Driver takes its ordinary lock itself. Wrapper lock remains held throughout.
    fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
    try:
        driver.execute(argparse.Namespace(device=args.device,suite=args.suite,out=root,threads=args.threads))
    finally:
        # Existing training/raw files must be byte-identical; new files are expected.
        changed=[]
        for rel,digest in manifest['preserved_input_sha256'].items():
            if not (root/rel).is_file() or sha(root/rel)!=digest:changed.append(rel)
        if changed:
            save_json(root/'numerical_audits'/'preservation_failure.json',{'changed':changed})
            raise RuntimeError('Beklenmeyen onceki kayit degisikligi: '+', '.join(changed[:5]))
        print('[KONTROL] Onceki ham olcumler ve tamamlanmis egitim dosyalari aynen korundu.',flush=True)
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path.home()/'mavi_inek8')
    parser.add_argument('--out',type=Path,default=Path.home()/'spektrum_mekanizma8')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--suite',choices=('smoke','sentinel','mechanism'),default='mechanism')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--check-only',action='store_true',help='Yalniz sayisal ve kayitli veri kontrolleri; egitim yok.')
    args=parser.parse_args()
    if not 1<=args.threads<=32:parser.error('--threads 1..32 olmali')
    try:return run(args)
    except (Exception,KeyboardInterrupt) as exc:
        root=args.out.expanduser()
        report={'status':'FAILED','revision':REVISION,'error':str(exc),'traceback':traceback.format_exc()}
        if root.is_dir():save_json(root/'numerical_fix_v2_error.json',report)
        print('HATA: '+str(exc),file=sys.stderr,flush=True);traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
