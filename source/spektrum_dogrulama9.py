#!/usr/bin/env python3
"""Experiment 9: frozen L=8 validation of experiment-8 structure estimators.
Only PyTorch + standard library. CPU small-model fitting, optional CUDA physics.
No edits to older experiments or installed packages. Same command resumes.
The new matrix-free reference is checked against dense L4/L6; it keeps explicit
source/ground and response convergence diagnostics. No quantum advantage claim.
Pilot uses seed 9100, confirmation 9200:9219; all method settings are frozen.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,time,traceback,zipfile,fcntl
from pathlib import Path
import torch
import hubbard_mf9 as mf
import numerical8_verified as stable
RD=torch.float64;CD=torch.complex128
VERSION='0.9.0-L8-frozen-validation'
PROJECT=Path(__file__).resolve().parent
SOURCE_FILES=('spektrum_dogrulama9.py','hubbard_mf9.py','numerical8_verified.py','core6.py','spektrum_mekanizma8.py','spektrum_rakip8b.py')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def matpack(x):return {'real':x.real.detach().cpu().tolist(),'imag':x.imag.detach().cpu().tolist()}
def matread(x):return torch.complex(torch.tensor(x['real'],dtype=RD),torch.tensor(x['imag'],dtype=RD))
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def csvwrite(p,rows):
    p=Path(p);tmp=p.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(p)

def physics_checks(core,device):
    result={}
    for L,U in ((4,4.),(6,7.)):
        p=mf.make_problem(L,U,device);old=core.build_problem(L,U,device)
        require(abs(p['E0']-old['E0'])<1e-8,'Matrix-free ground reference mismatch')
        om=torch.linspace(-12,12,241,dtype=RD)
        errors=[]
        for br,bb in zip(p['branches'],old['branches']):
            E,w,ra=mf.response(br['K'],br['b'],om)
            target=-core.green(bb['K'],bb['b'],(om+.2j).to(device)).imag.cpu()/math.pi
            error=float((mf.spectral_poles(E,w,om)-target).abs().max())
            require(error<1e-8,'Matrix-free spectral reference mismatch');errors.append(error)
            data,_,_=core.fixed_time_pool(L,U,br,24,.5,.125)
            ref,_,_=core.fixed_time_pool(L,U,bb,24,.5,.125)
            require(float((data['S']-ref['S']).abs().max())<1e-8,'Time-pool overlap mismatch')
            require(float((data['T']-ref['T']).abs().max())<1e-8,'Time-pool energy mismatch')
        result[f'L{L}']={'ground_error':abs(p['E0']-old['E0']),'spectral_errors':errors}
    return result

def prepare_reference(core,root,protocol,ph,device,log):
    path=root/'reference_cache.json'
    if path.exists():
        cache=json.loads(path.read_text());require(cache['protocol_sha256']==ph,'Reference cache protocol mismatch')
        log('[FIZIK] Kayitli sayisal referans ve olcum beklentileri kullaniliyor.');return cache
    L,U=protocol['L'],protocol['U'];log(f'[FIZIK] L={L}: buyuk Hamiltonian matrisi kurulmadan referans hesaplaniyor.')
    with torch.no_grad():
        prob=mf.make_problem(L,U,device);omega=torch.linspace(-12,12,2401,dtype=RD)
        wp=torch.linspace(-math.pi/.5,math.pi/.5,2401,dtype=RD)
        layout=core.SettingLayout(core.time_indices6(24),torch.device('cpu'))
        branches=[];truth=torch.zeros_like(omega);period=torch.zeros_like(wp)
        for br in prob['branches']:
            E,w,audit=mf.response(br['K'],br['b'],omega)
            truth+=mf.spectral_poles(E,w,omega);period+=mf.periodic_poles(E,w,wp)
            branches.append({'E':E.tolist(),'weights':w.tolist(),'audit':audit})
        backends={}
        for step in protocol['steps']:
            entries=[];core.PRODUCT_STEP=step
            for bid,br in enumerate(prob['branches']):
                data,recipes,_=core.fixed_time_pool(L,U,br,24,.5,step)
                back=core.SamplingBackend(data,br,prob,layout,bid,f'L8_validation9_step{step}')
                entries.append({'data':{k:matpack(v) for k,v in data.items()},'info':back.info,
                     'means':back._means.tolist(),'sampling_label':back._label,'recipes':recipes})
            backends[str(step)]=entries
        cache={'protocol_sha256':ph,'ground':prob['ground_audit'],'E0':prob['E0'],
            'L':L,'U':U,'branches':branches,'reference_nonperiodic':truth.tolist(),
            'reference_periodic':period.tolist(),'omega':omega.tolist(),'omega_periodic':wp.tolist(),
            'backends':backends,'note':'Ideal simulation internals; never inputs to model selection.'}
    dump(path,cache);log('[FIZIK] Kaynak ve spektrum referansi kontrol edilip kaydedildi.');return cache

class CachedSampler:
    """Private simulator probabilities; estimator only receives sampled raw data."""
    def __init__(self,entry):
        self._means=torch.tensor(entry['means'],dtype=RD);self._label=entry['sampling_label'];self.info=entry['info']
    def sample(self,counts,seed,stage):
        require(counts.shape==self._means.shape and int(counts.min())>0,'Invalid shot allocation')
        token=f'{self._label}|seed={seed}|stage={stage}'
        gen=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'little')%(2**63-1))
        return 2*torch.binomial(counts.to(RD),(1+self._means)/2,generator=gen)/counts-1
    def overlap_budget(self,ns,shots,seed):
        token=f'{self._label}|seed={seed}|stage=time_only_full_budget'
        gen=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'little')%(2**63-1))
        n=torch.full((ns,),shots,dtype=RD)
        return 2*torch.binomial(n,(1+self._means[:ns])/2,generator=gen)/shots-1

def execute(args):
    torch.set_num_threads(args.threads)
    root=args.out.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    def log(s):
        print(s,flush=True)
        with (root/'run.log').open('a',encoding='utf8') as f:f.write(s+'\n')
    driver=stable.load_driver(PROJECT);core=driver.core;rival=driver.rival
    adapter=stable.install_local_adapters(driver)
    device=core.get_device(args.device);tests={'cpu_arithmetic':stable.arithmetic_audit('cpu')}
    if device.type=='cuda':tests['gpu_arithmetic']=stable.arithmetic_audit(str(device))
    with torch.no_grad():tests['time_reconstruction']=rival.self_tests8b();tests['physics']=physics_checks(core,device)
    tests['snapshot_contract']=driver.test_contract();tests['module_local_numerics']=adapter
    dump(root/'self_tests.json',tests);dump(root/'environment.json',core.environment(device))
    log('Dogrulama: PASS; CPU karmasik dot korumasi ve matris-kurmadan fizik testleri gecti.')
    smoke=args.suite=='smoke';L,U=(4,4.) if smoke else (8,7.)
    steps=[.125] if smoke else [.125,.25];budgets=[10**8] if smoke else [10**8,10**9]
    seeds=[9199] if smoke else ([9100] if args.suite=='pilot' else list(range(9200,9220)))
    ranks=[6] if smoke else [6,10,14];starts=1 if smoke else 2
    milestones=[0,10,20] if smoke else [0,100,350,700,1400]
    protocol={'version':VERSION,'suite':args.suite,'L':L,'U':U,'steps':steps,'budgets':budgets,'seeds':seeds,
        'ranks':ranks,'starts':starts,'milestones':milestones,'training_fraction':.8,'refit_after_selection':False,
        'eta':.2,'spacing':.5,'input_states_per_branch':24,'same_estimators_as_experiment8':True,
        'source_and_E0':'Converged numerical ground/source; preparation cost excluded; ideal gates.',
        'selection':'Held-out binomial loss only; no spectral truth.',
        'time_only_controls':['same archived overlaps','all B reassigned to 46 overlap settings per branch'],
        'time_metric':'periodic physical spectrum only; never interpreted as unaliased full spectrum',
        'pilot_not_confirmation':True,'hashes':{n:sha(PROJECT/n) for n in SOURCE_FILES}}
    ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (root/'protocol.json').exists():require(json.loads((root/'protocol.json').read_text())['protocol_sha256']==ph,'Protocol changed; choose a new --out')
    dump(root/'protocol.json',dict(protocol,protocol_sha256=ph))
    for sub in ('raw','fits','conditions','curves'):(root/sub).mkdir(exist_ok=True)
    cache=prepare_reference(core,root,protocol,ph,device,log)
    layout=core.SettingLayout(core.time_indices6(24),torch.device('cpu'))
    omega=torch.tensor(cache['omega'],dtype=RD);z=omega+.2j;wp=torch.tensor(cache['omega_periodic'],dtype=RD)
    total=len(steps)*len(budgets)*len(seeds);rows=[];counter=0;clock=time.perf_counter()
    log(f'Plan: {total} kosul; L={L}; egitim CPU, fizik {device}; ayarlar Deney 8 ile ayni.')
    for step in steps:
      core.PRODUCT_STEP=step
      backends=[CachedSampler(e) for e in cache['backends'][str(step)]]
      for B in budgets:
       for seed in seeds:
        counter+=1;tag=f'L{L}_U{U:g}_step{step:g}_B{B}_seed{seed}';cp=root/'conditions'/(tag+'.json')
        if cp.exists():
            c=json.loads(cp.read_text());require(c['protocol_sha256']==ph,'Checkpoint changed')
            rows.extend(c['rows']);log(f'[{counter}/{total}] {tag}: tamam, atlandi.');continue
        log(f'[{counter}/{total}] {tag}');raws=[];choices=[];greens={'raw_cutoff':[],'Hlag_cutoff':[]}
        for bid,back in enumerate(backends):
            cn=torch.full((layout.nsettings,),B//(2*layout.nsettings),dtype=torch.int64)
            ct=torch.floor(.8*cn.to(RD)).to(torch.int64);cv=cn-ct
            rawpath=root/'raw'/(tag+f'_branch{bid}.json')
            if rawpath.exists():
                raw=json.loads(rawpath.read_text());require(raw['protocol_sha256']==ph,'Raw protocol changed')
            else:
                raw={'protocol_sha256':ph,'keys':layout.keys,'q':layout.q,'info':back.info,
                    'counts_train':ct.tolist(),'counts_validation':cv.tolist(),
                    'observed_train':back.sample(ct,seed,'validation9_train').tolist(),
                    'observed_validation':back.sample(cv,seed,'validation9_selection').tolist()}
                dump(rawpath,raw)
            ch=driver.fit_branch(raw,layout,root/'fits',tag+f'_branch{bid}',ph,milestones,ranks,starts,log)
            choices.append(ch);raws.append(raw)
            with torch.no_grad():
                y=(ct*torch.tensor(raw['observed_train'],dtype=RD)+cv*torch.tensor(raw['observed_validation'],dtype=RD))/cn
                S,T=layout.matrices(y,back.info);H=(back.info['sign']*T+back.info['shift']*S)/back.info['scale']
                Hpool=rival.lag_projection(H,layout.q);Tpool=back.info['sign']*(back.info['scale']*Hpool-back.info['shift']*S)
                for name,t in [('raw_cutoff',T),('Hlag_cutoff',Tpool)]:greens[name].append(core.project_measured(S,t,back.info['beta'],z,int(cn[0]))[0])
        # All model choices are frozen before evaluation accesses reference spectra.
        with torch.no_grad():
            specs={name:-sum(g).imag/math.pi for name,g in greens.items()};pspecs={}
            for name in ('diagonal_cv','general_cv_including_diagonal'):
                gg=[];pp=[]
                for ch,raw in zip(choices,raws):
                    choice=ch[name];model=driver.restore(layout,choice['model'],choice['diagonal'])
                    gg.append(model.green(raw['info'],z));pp.append(rival.latent_periodic(choice['model'],raw['info'],wp,.5))
                specs[name]=-sum(gg).imag/math.pi;pspecs[name]=sum(pp)
            controls={'time_same_data':[],'time_full_budget':[]};controlcost={k:0 for k in controls};controlshots={k:0 for k in controls}
            nr=round(.5/step)*(4*(L-1)+L)
            allcost=2*int((cn*layout.time_tick_cost).sum())*nr
            for bid,(raw,back) in enumerate(zip(raws,backends)):
                f,_,_=rival.read_archived_f(raw);sign=raw['info']['sign']
                phase=torch.exp(1j*sign*(cache['E0']+U*(L+sign)/2)*.5*torch.arange(24,dtype=RD))
                ntime=B//(2*layout.ns);ym=back.overlap_budget(layout.ns,ntime,seed)
                fnew=torch.ones(24,dtype=CD);fnew[1:]=torch.complex(ym[0::2],ym[1::2])
                for name,ff in [('time_same_data',f),('time_full_budget',fnew)]:
                    val,au=rival.reconstruct_time(ff,phase,wp,.5,.2)
                    sp=val['psd_central_extension'];require(abs(float(core.integral(sp,wp))-1)<1e-7,'Time spectrum mass')
                    controls[name].append(raw['info']['beta']**2*sp)
                controlcost['time_full_budget']+=ntime*int(layout.time_tick_cost[:layout.ns].sum())*nr
                controlshots['time_full_budget']+=ntime*layout.ns
            pspecs.update({name:sum(v) for name,v in controls.items()})
            truth=torch.tensor(cache['reference_nonperiodic'],dtype=RD);ptruth=torch.tensor(cache['reference_periodic'],dtype=RD)
            mass=float(core.integral(truth,omega));rr=[]
            for name in sorted(set(specs)|set(pspecs)):
                if name in specs:require(bool(torch.isfinite(specs[name]).all()) and float(specs[name].min())>=-1e-8,'Invalid spectrum')
                rr.append({'L':L,'U':U,'step':step,'budget':B,'seed':seed,'method':name,
                    'nonperiodic_relative_L1':float(core.integral(abs(specs[name]-truth),omega))/mass if name in specs else None,
                    'periodic_L1':float(core.integral(abs(pspecs[name]-ptruth),wp)) if name in pspecs else None,
                    'actual_shots':controlshots[name] if name=='time_full_budget' else 2*int(cn.sum()),
                    'shot_weighted_rotation_proxy':controlcost[name] if name=='time_full_budget' else allcost})
            require(all(r['actual_shots']<=B for r in rr),'Total budget exceeded')
            csvwrite(root/'curves'/(tag+'.csv'),[{'omega':float(omega[i]),'exact':float(truth[i]),**{n:float(v[i]) for n,v in specs.items()}} for i in range(len(omega))])
            csvwrite(root/'curves'/(tag+'_periodic.csv'),[{'omega':float(wp[i]),'exact':float(ptruth[i]),**{n:float(v[i]) for n,v in pspecs.items()}} for i in range(len(wp))])
            dump(cp,{'protocol_sha256':ph,'rows':rr,'selected':choices});rows.extend(rr);csvwrite(root/'metrics.csv',rows)
            log('  Hatalar: '+'; '.join(f'{r["method"]}=%{100*r["nonperiodic_relative_L1"]:.3f}' for r in rr if r['nonperiodic_relative_L1'] is not None))
    csvwrite(root/'metrics.csv',rows)
    dump(root/'summary.json',{'status':'PASS','conditions_completed':total,'rows':len(rows),'protocol_sha256':ph,
        'seconds_this_session':time.perf_counter()-clock,'new_quantum_hardware':False,
        'interpretation':'numerical simulation; pilot results are not confirmation; source/gates ideal'})
    log(f'SONUC: PASS — {total}/{total}; yayin veya ustunluk garantisi DEGIL.')
    (root/'source').mkdir(exist_ok=True)
    for name in SOURCE_FILES:(root/'source'/name).write_bytes((PROJECT/name).read_bytes())
    archive=root.parent/(root.name+'_rapor.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zz:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name!='.lock':zz.write(p,p.relative_to(root))
    log('RAPOR_ZIP: '+str(archive))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda');p.add_argument('--suite',choices=('smoke','pilot','validation'),default='validation')
    p.add_argument('--out',type=Path,default=Path.home()/'spektrum_dogrulama9');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.threads<=16:p.error('threads 1..16 olmali')
    try:execute(a)
    except (Exception,KeyboardInterrupt) as e:
        print('HATA:',e,flush=True);traceback.print_exc();return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
