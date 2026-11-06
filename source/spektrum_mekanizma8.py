#!/usr/bin/env python3
"""Controlled experiment 8C: finite-shot structural bias at FIXED circuit depth.

Full suite: L6/U7; product steps .125/.25; B=1e7/1e8/1e9/1e10;
20 independent seeds 8100:8119, hence 160 conditions / 320 branches.
Noise distribution, exact sources and ideal E0 are inherited from experiment6.
Do not interpret this as general quantum advantage or native-hardware cost.

Both latent families train ONLY on 80% of the counted measurements.
Rank/start/early checkpoint selection uses the separate 20% selection part.
The general family includes ALL diagonal candidates, then releases the best
TRAINING-likelihood diagonal snapshot of each start. No true spectrum is used.
The selected checkpoint is NOT refitted on combined data (avoids erasing the
selected stopping rule). Classical costs are NOT equal across model families.

Main structural comparison: original physical [-12,12] spectral L1.
Time-only comparator: separate one-period physical metric (aliasing explicit).
Complements, does not replace, the stronger fine-step resource check in 8B.
Checkpointed after each training block; same command safely resumes.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,statistics,sys,time,traceback,zipfile
from pathlib import Path
import torch
import core6 as core
import spektrum_rakip8b as rival
VERSION='0.8c.0-holdout-controlled-structure'
CDTYPE=torch.complex128
RDTYPE=torch.float64


def snapshot(model):
    with torch.no_grad():
        w,V,A=model.core()
        return {'phases':model.theta.tolist(),'weights':w.tolist(),
                'energy_real':A.real.tolist(),'energy_imag':A.imag.tolist()}


def restore(layout,saved,diagonal):
    theta=torch.tensor(saved['phases'],dtype=RDTYPE);w=torch.tensor(saved['weights'],dtype=RDTYPE)
    A=torch.complex(torch.tensor(saved['energy_real'],dtype=RDTYPE),torch.tensor(saved['energy_imag'],dtype=RDTYPE))
    model=core.LatentPhysicalModel(layout,len(w),theta,w,A,diagonal)
    with torch.no_grad():
        model.logits.copy_(w.clamp_min(1e-300).log())
        if diagonal:model.energy_raw.copy_(torch.atanh(A.diag().real.clamp(-1+1e-13,1-1e-13)))
        else:
            e,V=torch.linalg.eigh((A+A.mH)/2);e=e.clamp(-1+1e-13,1-1e-13)
            R=((1+e)/(1-e)).sqrt()[:,None]*V.mH
            model.energy_raw.copy_(torch.view_as_real(R).contiguous())
        core.check(float((model.core()[2]-A).abs().max())<1e-8,'Saved-model restore mismatch.')
    return model


def perturb(model,token):
    seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'little')%(2**63-1)
    gen=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.theta.add_(.015*torch.randn(model.theta.shape,generator=gen,dtype=RDTYPE))
        model.logits.add_(.03*torch.randn(model.logits.shape,generator=gen,dtype=RDTYPE))
        model.energy_raw.add_(.015*torch.randn(model.energy_raw.shape,generator=gen,dtype=RDTYPE))


def train_block(model,y,n,steps):
    """Optimizer restarts at predeclared blocks, identically on resumption."""
    begin=time.perf_counter()
    audit=core.train_latent(model,y,n,steps)
    with torch.enable_grad():
        for p in model.parameters():p.grad=None
        (1e4*core.binomial_deviance(model(),y,n)).backward()
        audit['fresh_scaled_gradient_max']=max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None)
    audit['seconds']=time.perf_counter()-begin
    return audit


def select_candidate(records):
    # Fixed tie-break favors the smaller model; spectral truth is not an input.
    return min(records,key=lambda r:(r['selection_deviance'],r['rank'],r['nominal_iterations'],r['name']))


def fit_branch(raw,layout,cpdir,tag,protocol,milestones,ranks,starts,log):
    yt=torch.tensor(raw['observed_train'],dtype=RDTYPE);yv=torch.tensor(raw['observed_validation'],dtype=RDTYPE)
    ct=torch.tensor(raw['counts_train'],dtype=torch.int64);cv=torch.tensor(raw['counts_validation'],dtype=torch.int64)
    info=raw['info'];diagonals=[];generals=[]
    rawhash=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
    def track(initial,diagonal,name,rank):
        path=cpdir/f'{tag}_{name}.json'
        if path.exists():
            saved=json.loads(path.read_text());core.check(saved['protocol_sha256']==protocol and saved['raw_sha256']==rawhash,'Checkpoint protocol/raw mismatch.')
            records=saved['records'];model=restore(layout,records[-1]['model'],diagonal)
        else:
            model=initial();records=[]
        for milestone in milestones:
            if any(r['nominal_iterations']==milestone for r in records):continue
            prev=records[-1]['nominal_iterations'] if records else 0
            fit={} if milestone==0 else train_block(model,yt,ct,milestone-prev)
            with torch.no_grad():
                record={'name':name,'rank':rank,'nominal_iterations':milestone,'fit':fit,
                    'training_deviance':float(core.binomial_deviance(model(),yt,ct)),
                    'selection_deviance':float(core.binomial_deviance(model(),yv,cv)),
                    'model':snapshot(model),'diagonal':diagonal}
            records.append(record)
            core.dump_json(path,{'protocol_sha256':protocol,'raw_sha256':rawhash,'records':records})
        log(f'    {tag} {name}: egitim bloklari kayitli.')
        return records
    for rank in ranks:
        for start in range(starts):
            def init_diag():
                model=core.initialize_latent(yt,layout,info,rank,True)
                if start:perturb(model,f'{tag}|{rank}|{start}')
                return model
            dd=track(init_diag,True,f'diag_r{rank}_s{start}',rank);diagonals.extend(dd)
            # General family can choose a constrained candidate; never handicap it.
            generals.extend([dict(r,name='included_'+r['name']) for r in dd])
            training_best=min(dd,key=lambda r:r['training_deviance'])
            def init_general():return restore(layout,training_best['model'],False)
            gg=track(init_general,False,f'general_r{rank}_s{start}',rank);generals.extend(gg)
    diagonal=select_candidate(diagonals);general=select_candidate(generals)
    core.check(general['selection_deviance']<=diagonal['selection_deviance']+1e-14,
               'General candidate set did not include diagonal family.')
    return {'diagonal_cv':diagonal,'general_cv_including_diagonal':general,
        'candidate_counts':{'diagonal':len(diagonals),'general':len(generals)},
        'selection_uses_true_spectrum':False,'refitted_after_selection':False}


def test_contract():
    records=[{'selection_deviance':.2,'rank':6,'nominal_iterations':0,'name':'a'},
             {'selection_deviance':.1,'rank':10,'nominal_iterations':10,'name':'b'}]
    core.check(select_candidate(records)['name']=='b','Selection must use held-out likelihood.')
    layout=core.SettingLayout([0,1,-1,2,-2],torch.device('cpu'))
    saved={'phases':[-.5,.7],'weights':[.4,.6],'energy_real':[[-.991,0],[0,.993]],'energy_imag':[[0,0],[0,0]]}
    d=restore(layout,saved,True);g=restore(layout,saved,False)
    err=float((d()-g()).abs().max().detach());core.check(err<1e-11,'Nested model restoration failed.')
    return {'status':'PASS','selection_by_holdout_only':True,'nested_embedding_max_error':err}


def execute(args):
    import fcntl
    torch.set_num_threads(args.threads);device=core.get_device(args.device)
    root=args.out.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def log(s):
        print(s,flush=True)
        with (root/'run.log').open('a',encoding='utf8') as f:f.write(s+'\n')
    smoke=args.suite=='smoke';sentinel=args.suite=='sentinel'
    L,U=(4,4.) if smoke else (6,7.)
    steps=[.125] if smoke else [.125,.25];budgets=[10**8] if (smoke or sentinel) else [10**7,10**8,10**9,10**10]
    seeds=[8999] if smoke else ([8100] if sentinel else list(range(8100,8120)));ranks=[6] if smoke else [6,10,14]
    milestones=[0,10,20] if smoke else [0,100,350,700,1400];starts=1 if smoke else 2
    protocol={'version':VERSION,'suite':args.suite,'L':L,'U':U,'steps':steps,'budgets':budgets,'seeds':seeds,
        'ranks':ranks,'starts':starts,'milestones':milestones,'train_fraction':.8,
        'optimizer_restart_at_each_block':True,'main_metric':'physical nonperiodic L1 on [-12,12]',
        'rival_metric':'separate periodic physical L1; time-only spectrum not unfolded',
        'source_E0':'exact, common, preparation uncharged','eta':.2,'spacing':.5,
        'general_contains_all_diagonal_candidates':True,'no_refit_after_selection':True,
        'selection':'minimum held-out binomial deviance; no true spectrum',
        'maximum_depth':'fixed per product-step stratum; no unrestricted optimal-resource claim',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'core_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        'rival_sha256':hashlib.sha256(Path(rival.__file__).read_bytes()).hexdigest()}
    ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (root/'protocol.json').exists():
        core.check(json.loads((root/'protocol.json').read_text())['protocol_sha256']==ph,'Protocol changed; choose new --out.')
    core.dump_json(root/'protocol.json',dict(protocol,protocol_sha256=ph))
    core.dump_json(root/'environment.json',core.environment(device))
    tests=test_contract()
    with torch.no_grad():tests.update(core.self_tests(device))
    core.dump_json(root/'self_tests.json',tests);log('Dogrulama: PASS; bu bir deney sonucu DEGIL.')
    total=len(steps)*len(budgets)*len(seeds);log(f'Plan: {total} kosul; egitim CPU, fizik {device}.')
    for name in ('fits','raw','conditions','curves'):(root/name).mkdir(exist_ok=True)
    with torch.no_grad():problem=core.build_problem(L,U,device)
    layout=core.SettingLayout(core.time_indices6(24),torch.device('cpu'))
    omega=torch.linspace(-12,12,2401,dtype=RDTYPE);z=omega+.2j
    wperiod=torch.linspace(-math.pi/.5,math.pi/.5,2401,dtype=RDTYPE)
    rows=[];counter=0;clock=time.perf_counter()
    for step in steps:
        core.PRODUCT_STEP=step
        with torch.no_grad():
            backend=[]
            for bid,br in enumerate(problem['branches']):
                data,_,_=core.fixed_time_pool(L,U,br,24,.5,step)
                backend.append(core.SamplingBackend(data,br,problem,layout,bid,f'mechanism8_step{step}'))
        for B in budgets:
            for seed in seeds:
                counter+=1;tag=f'L{L}_U{U:g}_step{step:g}_B{B}_seed{seed}';cp=root/'conditions'/f'{tag}.json'
                if cp.exists():
                    old=json.loads(cp.read_text());core.check(old['protocol_sha256']==ph,'Case protocol mismatch.')
                    rows.extend(old['rows']);log(f'[{counter}/{total}] {tag}: tamam, atlandi.');continue
                log(f'[{counter}/{total}] {tag}')
                allchoices=[];raws=[];allinfo=[];baseline_green={'raw_cutoff':[],'Hlag_cutoff':[]}
                for bid,back in enumerate(backend):
                    rawpath=root/'raw'/f'{tag}_branch{bid}.json'
                    cn=torch.full((layout.nsettings,),B//(2*layout.nsettings),dtype=torch.int64)
                    ct=torch.floor(cn.to(RDTYPE)*.8).to(torch.int64);cv=cn-ct
                    if rawpath.exists():
                        raw=json.loads(rawpath.read_text());core.check(raw['protocol_sha256']==ph,'Raw protocol mismatch.')
                    else:
                        yt=back.sample(ct,seed,'mechanism8_train');yv=back.sample(cv,seed,'mechanism8_selection')
                        raw={'protocol_sha256':ph,'keys':layout.keys,'q':layout.q,'info':back.info,
                             'counts_train':ct.tolist(),'counts_validation':cv.tolist(),
                             'observed_train':yt.tolist(),'observed_validation':yv.tolist()}
                        core.dump_json(rawpath,raw)
                    choices=fit_branch(raw,layout,root/'fits',tag+f'_branch{bid}',ph,milestones,ranks,starts,log)
                    allchoices.append(choices);raws.append(raw);allinfo.append(back.info)
                    with torch.no_grad():
                        yt=torch.tensor(raw['observed_train'],dtype=RDTYPE);yv=torch.tensor(raw['observed_validation'],dtype=RDTYPE)
                        yy=(ct*yt+cv*yv)/cn;S,T=layout.matrices(yy,back.info)
                        G,_=core.project_measured(S,T,back.info['beta'],z,int(cn[0]));baseline_green['raw_cutoff'].append(G)
                        H=(back.info['sign']*T+back.info['shift']*S)/back.info['scale']
                        Hp=rival.lag_projection(H,layout.q)
                        Tp=back.info['sign']*(back.info['scale']*Hp-back.info['shift']*S)
                        G,_=core.project_measured(S,Tp,back.info['beta'],z,int(cn[0]));baseline_green['Hlag_cutoff'].append(G)
                # Evaluation starts AFTER ALL fitting and selection. Reference never enters fits.
                with torch.no_grad():
                    truth=-sum(core.green(br['K'],br['b'],z.to(device)) for br in problem['branches']).imag.cpu()/math.pi
                    mass=float(core.integral(truth,omega));ptruth=0
                    for br in problem['branches']:
                        e,V=torch.linalg.eigh(br['K'].cpu());w=(V.mH@br['b'].cpu()).abs().square()
                        ptruth+=rival.periodic_poles(e,w,wperiod,.5,.2)
                    specs={name:-sum(v).imag/math.pi for name,v in baseline_green.items()}
                    pspecs={}
                    for name in ('diagonal_cv','general_cv_including_diagonal'):
                        Gs=[];pspec=0
                        for choice,info in zip(allchoices,allinfo):
                            chosen=choice[name];model=restore(layout,chosen['model'],chosen['diagonal'])
                            Gs.append(model.green(info,z));pspec+=rival.latent_periodic(chosen['model'],info,wperiod,.5)
                        specs[name]=-sum(Gs).imag/math.pi;pspecs[name]=pspec
                    tspec=0
                    for bid,raw in enumerate(raws):
                        f,_,_=rival.read_archived_f(raw);sgn=raw['info']['sign']
                        gamma=problem['E0']+U*(L+sgn)/2
                        phase=torch.exp(1j*sgn*gamma*.5*torch.arange(24,dtype=RDTYPE))
                        val,_=rival.reconstruct_time(f,phase,wperiod,.5,.2)
                        tspec+=raw['info']['beta']**2*val['psd_central_extension']
                    pspecs['published_time_central_extension']=tspec
                    used=2*int(cn.sum());core.check(used<=B,'Budget exceeded.')
                    rr=[]
                    for name in sorted(set(specs)|set(pspecs)):
                        err=float(core.integral(abs(specs[name]-truth),omega))/mass if name in specs else None
                        perr=float(core.integral(abs(pspecs[name]-ptruth),wperiod)) if name in pspecs else None
                        rr.append({'L':L,'U':U,'step':step,'budget':B,'seed':seed,'method':name,
                             'nonperiodic_relative_L1':err,'periodic_L1':perr,'actual_shots':used,
                             'selection_no_spectral_truth':True,'refit_after_selection':False})
                    curve=[{'omega':float(omega[i]),'exact':float(truth[i]),**{k:float(v[i]) for k,v in specs.items()}} for i in range(len(omega))]
                    rival.write_csv(root/'curves'/f'{tag}.csv',curve)
                    core.dump_json(cp,{'protocol_sha256':ph,'rows':rr,'selected':allchoices})
                    rows.extend(rr);rival.write_csv(root/'metrics.csv',rows)
                    log('  Hatalar: '+'; '.join(f'{r["method"]}=%{100*r["nonperiodic_relative_L1"]:.3f}' for r in rr if r['nonperiodic_relative_L1'] is not None))
    core.dump_json(root/'summary.json',{'status':'PASS','conditions_completed':total,'rows':len(rows),
        'protocol_sha256':ph,'seconds_this_session':time.perf_counter()-clock,
        'scope':'smoke test only' if smoke else 'new controlled development; no pre-established superiority'})
    log(f'SONUC: PASS — {total}/{total}; yayin/ustunluk garantisi DEGIL.')
    for module in (Path(__file__),Path(core.__file__),Path(rival.__file__)):
        (root/module.name).write_bytes(module.read_bytes())
    archive=root.parent/(root.name+'_rapor.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zz:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name!='.lock':zz.write(p,p.relative_to(root))
    log('RAPOR_ZIP: '+str(archive))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device',default='cuda');p.add_argument('--suite',choices=('smoke','sentinel','mechanism'),default='mechanism')
    p.add_argument('--out',type=Path,default=Path.home()/'spektrum_mekanizma8');p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.threads<=32:p.error('threads must be 1..32')
    try:execute(args)
    except (Exception,KeyboardInterrupt) as e:
        print('HATA: '+str(e),file=sys.stderr);traceback.print_exc();return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
