#!/usr/bin/env python3
"""Experiment 6: joint latent-unitary / energy reconstruction (exploratory).
Python >=3.10; only PyTorch + standard library. No installation or network.
Physics kernels may use CUDA; small-matrix fitting deliberately uses CPU.

S_ab = v_a^dagger v_b; Hbar_ab/lambda = v_a^dagger A v_b;
v_a[j] = sqrt(w_j) exp(-i theta_j q_a), w>=0, sum(w)=1, -I<A<I.
The phases of the product-formula propagator are NOT identified with the
physical energy eigenvalues. General Hermitian A need not commute with phases.
Measured binomial counts, NOT spectra/exact eigenvalues, enter estimation.
Candidate ranks are selected using independent held-out measurements (20%).
All counts are paid for; refitting uses those same observations, not new shots.
Eta is fixed at 0.2; no trainable widths, ground-truth loss or oracle rank.

Baselines: existing cutoff solver; PSD Toeplitz Gram-only fit; joint PSD
Gram/energy fit; diagonal-energy latent model (commuting ablation).
PSD fits use count-weighted LS, not a faithful implementation of a specific
published algorithm. Latent fits use binomial likelihood, with finite LBFGS
iterations. No global minimum, statistical coverage, novelty, quantum speedup
or publication-readiness guarantee. Exact source/E0; no gate noise.

References / prior art (not claims of novelty):
https://doi.org/10.1103/PhysRevA.108.012618
https://doi.org/10.1103/PhysRevLett.132.160403
https://doi.org/10.1103/PhysRevResearch.5.L022001
https://arxiv.org/abs/2205.00094
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,platform,random,statistics,sys,time,traceback,uuid,zipfile
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
from typing import Any
try:
    import torch
except ImportError as exc:
    raise SystemExit('PyTorch bulunamadi; onceki deneylerle ayni Python kullanilmali.') from exc
VERSION='0.6.1-joint-latent-pilot'
CDTYPE=torch.complex128
RDTYPE=torch.float64
SPACING=.5
PRODUCT_STEP=.125
ETA=.2
NOISE_CUTOFF_FACTOR=2.
RANKS=(6,10,14)
FIT_STEPS=350
FIT_FRACTION=.80


def integral(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sum((y[..., 1:] + y[..., :-1]) * (x[1:] - x[:-1]) * 0.5, dim=-1)

def dump_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

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

def green(K: torch.Tensor, b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    ev, vec = torch.linalg.eigh((K + K.mH) * 0.5)
    weight = torch.abs(vec.mH @ b).square()
    return torch.sum(weight[:, None] / (z[None, :] - ev[:, None]), dim=0)

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

def apply_gate(v: torch.Tensor, gate: dict[str, Any], theta: float) -> torch.Tensor:
    out = v.clone()
    if gate["kind"] == "hop":
        a, b, s = gate["a"], gate["b"], gate["sign"]
        out[a] = math.cos(theta)*v[a] - 1j*math.sin(theta)*s*v[b]
        out[b] = math.cos(theta)*v[b] - 1j*math.sin(theta)*s*v[a]
    else:
        out[gate["mask"]] *= complex(math.cos(theta), -math.sin(theta))
    return out

def compressed(C: torch.Tensor, data: dict[str, Any], z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T = C.mH @ data["T"] @ C
    f = C.mH @ data["f"]
    ev, vec = torch.linalg.eigh((T+T.mH)*0.5)
    c = vec @ ((vec.mH @ f)[:, None] / (z[None, :]-ev[:, None]))
    return torch.sum(f.conj()[:, None]*c, dim=0), c

def krylov_history(K: torch.Tensor, b: torch.Tensor, mmax: int) -> list[torch.Tensor]:
    Q = (b/torch.linalg.vector_norm(b))[:, None]
    history = []
    for _ in range(mmax):
        history.append(Q.clone())
        if Q.shape[1] >= mmax:
            break
        v = K @ Q[:, -1]
        for _pass in range(2):
            v = v-Q @ (Q.mH @ v)
        norm = float(torch.linalg.vector_norm(v))
        if norm < 1e-10:
            break
        Q = torch.cat((Q, (v/norm)[:, None]), dim=1)
    return history

def self_tests(device: torch.device) -> dict[str, float]:
    out: dict[str, float] = {}
    p = build_problem(4, 4.0, device)
    out["L4_U4_physical_ground_energy_error"] = abs(p["physical_E0"]+1.953145308684558)
    check(out["L4_U4_physical_ground_energy_error"] < 1e-10, "Referans enerji testi basarisiz.")
    omega = torch.linspace(-12, 12, 241, device=device, dtype=RDTYPE)
    z = omega+0.2j
    free = build_problem(4, 0.0, device)
    exact = sum(green(b["K"], b["b"], z) for b in free["branches"])
    k = torch.arange(1, 5, dtype=RDTYPE, device=device)*math.pi/5
    weights = 2/5*torch.sin(k*(free["orbital"]+1)).square()
    analytic = torch.sum(weights[:, None]/(z[None, :]+2*torch.cos(k)[:, None]), dim=0)
    out["noninteracting_green_max_error"] = float(torch.max(torch.abs(exact-analytic)))
    check(out["noninteracting_green_max_error"] < 1e-9, "U=0 analitik spektrum testi basarisiz.")
    branch = p["branches"][0]
    K, b = branch["K"], branch["b"]
    Q = krylov_history(K, b, 4)[-1]
    zz = torch.tensor(0.7+0.2j, dtype=CDTYPE, device=device)
    T, f = Q.mH @ K @ Q, Q.mH @ b
    c = torch.linalg.solve(zz*torch.eye(4, device=device, dtype=CDTYPE)-T, f)
    r = b-zz*(Q@c)+K@(Q@c)
    F = (K@Q).mH@(K@Q)
    identity = torch.vdot(c, (F-T@T)@c).real
    out["residual_identity_abs_error"] = float(torch.abs(torch.vdot(r, r).real-identity))
    check(out["residual_identity_abs_error"] < 1e-9, "Artik ozdesligi testi basarisiz.")
    # Complete tiny-sector projection must reproduce the full reference.
    I = torch.eye(len(b), dtype=CDTYPE, device=device)
    reference = green(K, b, z)
    projected, _ = compressed(I, {"T": K, "f": b}, z)
    out["full_projection_max_error"] = float(torch.max(torch.abs(reference-projected)))
    check(out["full_projection_max_error"] < 1e-9, "Tam altuzay testi basarisiz.")
    if device.type == "cuda":
        cpu = green(K.cpu(), b.cpu(), z.cpu())
        out["cpu_cuda_green_max_difference"] = float(torch.max(torch.abs(cpu-reference.cpu())))
        check(out["cpu_cuda_green_max_difference"] < 1e-8, "CPU/CUDA tutarlilik testi basarisiz.")
    return out

def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def get_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        check(torch.cuda.is_available(), "CUDA kullanilamiyor; CPU'ya sessizce gecilmedi.")
        torch.cuda.init()
        index = device.index if device.index is not None else 0
        check(0 <= index < torch.cuda.device_count(), "GPU indeksi gecersiz.")
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
        torch.empty(1, device=device)  # Initialize the allocator before optional telemetry.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif device.type != "cpu":
        raise ValueError("Bu deney yalniz cpu veya cuda[:index] kabul eder.")
    return device

def environment(device: torch.device) -> dict[str, Any]:
    info = {"python": sys.version, "platform": platform.platform(),
            "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "device": str(device), "dtype": "complex128", "cpu_threads": torch.get_num_threads()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device.index)
        try:
            torch.cuda.reset_peak_memory_stats(device.index)
            free, total = torch.cuda.mem_get_info(device.index)
            info.update(reported_free_bytes=free, reported_total_bytes=total)
        except (RuntimeError, ValueError) as exc:
            info["optional_memory_telemetry_warning"] = str(exc)
    return info

def product_step(v: torch.Tensor, groups: tuple, U: float, dt: float,
                 branch_sign: int, rng: random.Random | None = None) -> torch.Tensor:
    """Second-order even/odd/onsite symmetric product, up to global phase.

    H=-sum hopping + U sum double occupation. Chemical potential and E0
    contribute only an irrelevant global phase in each fixed-number sector.
    rng!=None independently flips signs of each scheduled elementary angle.
    """
    even, odd, onsite = groups
    h = branch_sign * dt
    for gates, angle in ((even, -h/2), (odd, -h/2), (onsite, U*h),
                         (odd, -h/2), (even, -h/2)):
        for gate in gates:
            a = angle if rng is None else angle * rng.choice((-1, 1))
            v = apply_gate(v, gate, a)
    return v

def product_groups(L: int, branch: dict[str, Any]) -> tuple:
    gates = make_gates(L, branch['basis'], branch['b'].device)
    even = [g for g in gates if g['kind'] == 'hop' and g['label'][2] % 2 == 0]
    odd = [g for g in gates if g['kind'] == 'hop' and g['label'][2] % 2 == 1]
    onsite = [g for g in gates if g['label'][0] == 'double_occupation_phase']
    return even, odd, onsite

def fixed_time_pool(L: int, U: float, branch: dict[str, Any], size: int,
                    spacing: float, step: float | None,
                    scrambled_seed: int | None = None) -> tuple[dict, list, torch.Tensor]:
    b = branch['b'] / torch.linalg.vector_norm(branch['b'])
    states, recipes = [b], [{'time': 0., 'steps': 0, 'scheduled_rotations': 0}]
    exact = step is None
    if exact:
        check(scrambled_seed is None, 'Exact-time scrambled pool is undefined.')
        ev, vec = torch.linalg.eigh(branch['K'])
        coef = vec.mH @ b
        rotations_per_step = 0
    else:
        ns = round(spacing/step)
        check(ns >= 1 and abs(ns*step-spacing) < 1e-12, 'Spacing must be a multiple of product step.')
        groups = product_groups(L, branch)
        rotations_per_step = 2*len(groups[0]) + 2*len(groups[1]) + len(groups[2])
        states_by_sign = {1: b.clone(), -1: b.clone()}
        rngs = {sgn: random.Random(131071*scrambled_seed + (17 if sgn == 1 else 29))
                if scrambled_seed is not None else None for sgn in (1, -1)}
    k = 1
    while len(states) < size:
        for sgn in (1, -1):
            if len(states) >= size:
                break
            time_value = sgn*k*spacing
            if exact:
                v = vec @ (torch.exp(-1j*ev*time_value) * coef)
                nsteps, nrot = None, None
            else:
                v = states_by_sign[sgn]
                for _ in range(ns):
                    v = product_step(v, groups, U, sgn*step, branch['energy_sign'], rngs[sgn])
                states_by_sign[sgn] = v
                nsteps, nrot = k*ns, k*ns*rotations_per_step
            states.append(v.clone())
            recipes.append({'time': time_value, 'steps': nsteps,
                            'scheduled_rotations': nrot, 'scrambled_seed': scrambled_seed})
        k += 1
    P = torch.stack(states, dim=1)
    S, T = P.mH @ P, P.mH @ (branch['K'] @ P)
    check(float(torch.max(torch.abs(S.diag().real-1))) < 1e-9, 'Time pool norm drift.')
    data = {'S': (S+S.mH)/2, 'T': (T+T.mH)/2, 'f': P.mH @ branch['b']}
    return data, recipes, P

def fixed_green(data: dict, z: torch.Tensor, size: int,
                gram_rtol: float) -> tuple[torch.Tensor, dict]:
    """Matrix-only projected solve. Keep the exact source, then truncate
    the source-orthogonal Gram matrix. No full states/eigen-spectrum needed.
    Every input circuit counts as acquired even when numerical rank is lower.
    """
    S = data['S'][:size, :size]
    T = data['T'][:size, :size]
    f = data['f'][:size]
    s0 = float(S[0, 0].real)
    check(s0 > .99, 'Invalid normalized source.')
    C0 = torch.zeros((size, 1), dtype=CDTYPE, device=S.device)
    C0[0, 0] = 1/math.sqrt(s0)
    R = S[1:, 1:] - S[1:, :1] @ S[:1, 1:] / s0
    vals, vecs = torch.linalg.eigh((R+R.mH)/2)
    vmax = max(float(vals[-1]), 0.) if vals.numel() else 0.
    check(not vals.numel() or float(vals[0]) > -1e-9*max(1., vmax), 'Nonpositive Gram matrix beyond rounding.')
    keep = vals > max(vmax*gram_rtol, 1e-14)
    if bool(keep.any()):
        D = vecs[:, keep] / torch.sqrt(vals[keep])[None, :]
        C = torch.cat((C0, torch.cat((-S[:1, 1:] @ D / s0, D), dim=0)), dim=1)
        retained_condition = vmax/float(vals[keep][0])
    else:
        C, retained_condition = C0, 1.
    # Symmetric metric correction limits cancellation error from whitening.
    M = C.mH @ S @ C
    me, mv = torch.linalg.eigh((M+M.mH)/2)
    check(float(me.min()) > .9 and float(me.max()) < 1.1, 'Gram normalization unstable.')
    C = C @ ((mv*me.rsqrt()[None, :]) @ mv.mH)
    defect = float(torch.max(torch.abs(C.mH @ S @ C - torch.eye(C.shape[1], dtype=CDTYPE, device=S.device))))
    check(defect < 2e-6, f'Gram orthogonality defect: {defect:g}')
    Tp, fp = C.mH @ T @ C, C.mH @ f
    G = green(Tp, fp, z)
    mass = float(torch.linalg.vector_norm(fp).square())
    expected_mass = float(torch.abs(f[0]).square()/s0)
    check(abs(mass-expected_mass) < 2e-6, 'Source mass lost in Gram truncation.')
    return G, {'input_states': size, 'retained_rank': C.shape[1],
               'source_mass': mass, 'gram_rtol': gram_rtol,
               'retained_source_orthogonal_gram_condition': retained_condition,
               'orthogonality_defect': defect}

def extra_tests(device: torch.device) -> dict:
    prob = build_problem(4, 4., device)
    br = prob['branches'][0]
    b = br['b']/torch.linalg.vector_norm(br['b'])
    groups = product_groups(4, br)
    v = product_step(b, groups, 4., .1, br['energy_sign'])
    w = product_step(v, groups, 4., -.1, br['energy_sign'])
    inverse_error = float(torch.linalg.vector_norm(w-b))
    check(inverse_error < 1e-11, 'Symmetric product inverse test failed.')
    e, V = torch.linalg.eigh(br['K'])
    target = V @ (torch.exp(-1j*.4*e)*(V.mH@b))
    errors = []
    for nsteps in (4, 8, 16):
        v = b.clone()
        for _ in range(nsteps):
            v = product_step(v, groups, 4., .4/nsteps, br['energy_sign'])
        overlap = torch.vdot(target, v)
        phase = overlap/torch.abs(overlap)
        errors.append(float(torch.linalg.vector_norm(v-target*phase)))
    check(errors[1] < errors[0]/3 and errors[2] < errors[1]/3,
          'Second-order step convergence failed.')
    omega = torch.linspace(-12, 12, 241, dtype=RDTYPE, device=device)
    z = omega+.2j
    data, rec, P = fixed_time_pool(4, 4., br, 12, .5, .125)
    G, info = fixed_green(data, z, 12, 1e-8)
    # This L4 source-orthogonal subspace is well conditioned. Independent QR
    # evaluator must agree with the matrix-only solve when no modes are dropped.
    check(info['retained_rank'] == 12, 'Unexpected rank in independent QR test.')
    Q, _ = torch.linalg.qr(P, mode='reduced')
    ref = green(Q.mH@br['K']@Q, Q.mH@br['b'], z)
    err = float(torch.max(torch.abs(ref-G)))
    check(err < 1e-6, 'Gram/independent-QR projected Green mismatch.')
    _, rec2, P2 = fixed_time_pool(4, 4., br, 12, .5, .125, 11)
    check([x['scheduled_rotations'] for x in rec] == [x['scheduled_rotations'] for x in rec2],
          'Scrambled rotation counts mismatch.')
    check(float(torch.linalg.vector_norm(P-P2)) > .1, 'Scrambling did not change states.')
    return {'symmetric_product_inverse_error': inverse_error,
            'second_order_projective_errors': errors, 'gram_vs_QR_green_max_difference': err,
            'matched_scheduled_rotation_counts': True}

def whiten_source(S: torch.Tensor, shots: int | None) -> tuple[torch.Tensor,dict]:
    """Keep the source exactly in the retained span. Drop nonpositive/too-small
    modes of the noisy source-orthogonal Gram matrix. This is regularization,
    NOT a certificate that its projected spectrum is close to the true one.
    """
    m=S.shape[0]
    C0=torch.zeros((m,1),dtype=CDTYPE,device=S.device); C0[0,0]=1
    R=S[1:,1:]-S[1:,:1]@S[:1,1:]
    vals,vecs=torch.linalg.eigh((R+R.mH)/2)
    vmax=max(float(vals.max()),0.) if vals.numel() else 0.
    noise=0. if shots is None else NOISE_CUTOFF_FACTOR*m/math.sqrt(shots)
    threshold=max(vmax*1e-8,noise,1e-14)
    keep=vals>threshold
    if bool(keep.any()):
        D=vecs[:,keep]/torch.sqrt(vals[keep])[None,:]
        C=torch.cat((C0,torch.cat((-S[:1,1:]@D,D),dim=0)),dim=1)
        condition=vmax/float(vals[keep].min())
    else: C=C0; condition=1.
    defect=float((C.mH@S@C-torch.eye(C.shape[1],dtype=CDTYPE,device=S.device)).abs().max())
    check(defect<5e-5,'Regularized Gram normalization unstable.')
    return C,{'rank':C.shape[1],'cutoff':threshold,'negative_gram_modes':int((vals < -1e-10).sum()),
              'orthogonality_defect':defect,'retained_condition':condition}

def project_measured(S:torch.Tensor,T:torch.Tensor,beta:float,z:torch.Tensor,
                     shots:int|None) -> tuple[torch.Tensor,dict]:
    C,info=whiten_source(S,shots)
    f=beta*S[:,0]
    Tp=C.mH@T@C; fp=C.mH@f
    G=green(Tp,fp,z)
    mass=float(torch.linalg.vector_norm(fp).square())
    check(abs(mass-beta*beta)<2e-5,'Measured source mass inconsistency.')
    info['source_mass']=mass
    return G,info

class SettingLayout:
    """Public measurement geometry, without exact probabilities or spectra."""
    def __init__(self, q: list[int], device: torch.device):
        check(q[0] == 0 and len(set(q)) == len(q), 'Invalid source/time layout.')
        self.q = q.copy()
        self.m = len(q)
        self.keys: list[tuple] = []
        overlap_directions, h_directions, costs = [], [], []
        lags = sorted({abs(i-j) for i in q for j in q if i != j})
        for d in lags:
            for part in ('R', 'I'):
                E = torch.zeros((self.m, self.m), dtype=CDTYPE)
                for i in range(self.m):
                    for j in range(self.m):
                        if q[j]-q[i] == d:
                            E[i,j] = 1 if part == 'R' else 1j
                        elif q[j]-q[i] == -d:
                            E[i,j] = 1 if part == 'R' else -1j
                self.keys.append(('S', d, part))
                overlap_directions.append(E)
                costs.append(abs(d))
        self.ns = len(self.keys)
        for i in range(self.m):
            for j in range(i, self.m):
                for part in (('R',) if i == j else ('R', 'I')):
                    E = torch.zeros((self.m, self.m), dtype=CDTYPE)
                    E[i,j] = 1 if part == 'R' else 1j
                    E[j,i] = E[i,j].conj()
                    self.keys.append(('H', i, j, part))
                    h_directions.append(E)
                    costs.append(abs(q[i])+abs(q[j]))
        self.nsettings = len(self.keys)
        self.ES = torch.stack(overlap_directions).to(device)
        self.EH = torch.stack(h_directions).to(device)
        self.time_tick_cost = torch.tensor(costs, dtype=torch.int64)
        self.device = device

    def matrices(self, observations: torch.Tensor, info: dict) -> tuple[torch.Tensor,torch.Tensor]:
        mu = observations.to(device=self.device, dtype=CDTYPE)
        S = torch.eye(self.m, dtype=CDTYPE, device=self.device)
        S = S + torch.einsum('k,kij->ij', mu[:self.ns], self.ES)
        Hbar = info['scale'] * torch.einsum('k,kij->ij', mu[self.ns:], self.EH)
        return S, info['sign'] * (Hbar - info['shift']*S)

class SamplingBackend:
    """Private classical simulator. A policy receives sample(), not _means.

    The +/-1 normalized Hbar random-Pauli Hadamard distribution is the same
    physical measurement model as experiment 4. A vectorized binomial draw
    samples independent settings; S-lag reuse is applied only when assembling
    matrices. Distinct stages use independent random streams.
    """
    def __init__(self, data: dict, branch: dict, problem: dict,
                 layout: SettingLayout, branch_id: int, grid_name: str):
        self.layout = layout
        L, U = problem['L'], problem['U']
        self.info = {'scale': 2*(L-1)+U*L/4,
                     'shift': problem['E0']+U*L/4,
                     'sign': branch['energy_sign'],
                     'beta': float(torch.linalg.vector_norm(branch['b']))}
        S = data['S'].detach().cpu()
        Hbar = self.info['sign']*data['T'].detach().cpu() + self.info['shift']*S
        means = []
        for key in layout.keys:
            if key[0] == 'S':
                i,j = next((i,j) for i in range(layout.m) for j in range(layout.m)
                           if layout.q[j]-layout.q[i] == key[1])
                val = complex(S[i,j])
            else:
                val = complex(Hbar[key[1],key[2]]) / self.info['scale']
            means.append(val.real if key[-1] == 'R' else val.imag)
        self._means = torch.tensor(means, dtype=RDTYPE)
        check(float(self._means.abs().max()) <= 1+1e-8, 'Outcome mean outside [-1,1].')
        self._means.clamp_(-1,1)
        self._label = f'{L}|{U:.12g}|{branch_id}|{grid_name}'

    def sample(self, counts: torch.Tensor, seed: int, stage: str) -> torch.Tensor:
        counts = counts.detach().cpu().to(torch.int64)
        check(counts.shape == self._means.shape and int(counts.min()) > 0,
              'Invalid per-setting allocation.')
        token = f'{self._label}|seed={seed}|stage={stage}'
        digest = hashlib.sha256(token.encode()).digest()
        generator = torch.Generator(device='cpu').manual_seed(
            int.from_bytes(digest[:8], 'little') % (2**63-1))
        hits = torch.binomial(counts.to(RDTYPE), (1+self._means)/2,
                              generator=generator)
        return 2*hits/counts-1

# ---------------------------------------------------------------------------
# Independent real-valued observation coordinates: one coordinate per setting.
# ---------------------------------------------------------------------------
def measurements_of(S:torch.Tensor, Hn:torch.Tensor, layout:SettingLayout) -> torch.Tensor:
    values=[]
    for key in layout.keys:
        if key[0]=='S':
            i,j=next((i,j) for i in range(layout.m) for j in range(layout.m)
                     if layout.q[j]-layout.q[i]==key[1])
            v=S[i,j]
        else: v=Hn[key[1],key[2]]
        values.append(v.real if key[-1]=='R' else v.imag)
    return torch.stack(values)


def cone_fit(y:torch.Tensor, counts:torch.Tensor, layout:SettingLayout,
             joint:bool, tol:float=1e-8, max_iter:int=6000) -> tuple[torch.Tensor,dict]:
    """Convex count-weighted projection via ADMM; finite numerical tolerance.
    Joint constraints: S+Hbar/lambda >=0, S-Hbar/lambda >=0.
    S is unit-diagonal Toeplitz; Hbar is Hermitian, NOT assumed Toeplitz.
    """
    m,ns=layout.m,layout.ns;device=layout.device
    E=torch.cat((layout.ES,layout.EH)) if joint else layout.ES
    yw=y.to(device) if joint else y[:ns].to(device)
    cw=counts if joint else counts[:ns]
    weights=cw.to(device,RDTYPE)/cw.to(device,RDTYPE).median()
    D=(2 if joint else 1)*E.abs().square().sum((1,2))
    I=torch.eye(m,device=device,dtype=CDTYPE)
    constant=I.expand(2 if joint else 1,-1,-1).clone()
    def affine(x):
        S=I+torch.einsum('k,kij->ij',x[:ns].to(CDTYPE),E[:ns])
        if not joint:return S[None]
        Hn=torch.einsum('k,kij->ij',x[ns:].to(CDTYPE),E[ns:])
        return torch.stack((S+Hn,S-Hn))
    def adj(Q):
        if not joint:return torch.einsum('kij,ij->k',E.conj(),Q[0]).real
        return torch.cat((torch.einsum('kij,ij->k',E[:ns].conj(),Q[0]+Q[1]).real,
                          torch.einsum('kij,ij->k',E[ns:].conj(),Q[0]-Q[1]).real))
    def positive(Q):
        ev,V=torch.linalg.eigh((Q+Q.mH)/2)
        return (V*ev.clamp_min(0)[:,None,:])@V.mH
    x=yw.clone();Z=positive(affine(x));dualvar=torch.zeros_like(Z);rho=1.
    primal=dual=float('inf')
    for it in range(1,max_iter+1):
        x=(weights*yw+rho*adj(Z-dualvar-constant))/(weights+rho*D)
        AX=affine(x);old=Z;Z=positive(AX+dualvar);dualvar+=AX-Z
        if it%25==0 or it==max_iter:
            primal=float(torch.linalg.vector_norm(AX-Z))
            dual=float(rho*torch.linalg.vector_norm(adj(Z-old)))
            if max(primal,dual)<tol:break
            if primal>10*dual:rho*=2;dualvar/=2
            elif dual>10*primal:rho/=2;dualvar*=2
    info={'iterations':it,'primal_residual':primal,'dual_residual':dual,
          'converged':max(primal,dual)<tol,
          'minimum_constraint_eigenvalue':float(torch.linalg.eigvalsh(affine(x)).min()),
          'weighted_ls_objective':float((weights*(x-yw).square()).sum()/2)}
    check(info['converged'],'Convex fit did not converge; no silent successful result.')
    return x if joint else torch.cat((x,y[ns:].to(device))),info


def binomial_deviance(pred:torch.Tensor, observed:torch.Tensor,
                      counts:torch.Tensor) -> torch.Tensor:
    """Per-shot KL(observed Bernoulli || predicted Bernoulli).
    Saturated-model constant removed. Same minimizer as exact binomial NLL.
    Tiny probability clipping is numerical, not an uncertainty guarantee.
    """
    p=((1+pred)/2).clamp(1e-12,1-1e-12)
    q=((1+observed.to(pred.device))/2).clamp(0,1)
    entropy=torch.where(q>0,q*torch.log(q.clamp_min(1e-300)),torch.zeros_like(q))
    entropy+=torch.where(q<1,(1-q)*torch.log1p(-q.clamp_max(1-1e-15)),torch.zeros_like(q))
    dev=entropy-q*torch.log(p)-(1-q)*torch.log1p(-p)
    n=counts.to(device=pred.device,dtype=RDTYPE)
    return (n*dev).sum()/n.sum()


class LatentPhysicalModel(torch.nn.Module):
    """Unitary phase model + a separate Hermitian contraction for energy.
    General A uses A=I-2(I+R^dagger R)^-1; thus -I<=A<I by construction.
    Diagonal A is an ablation; it forces energy and time propagator to commute.
    """
    def __init__(self, layout:SettingLayout, rank:int, theta:torch.Tensor,
                 weights:torch.Tensor, energy:torch.Tensor, diagonal:bool):
        super().__init__();self.layout=layout;self.rank=rank;self.diagonal=diagonal
        self.theta=torch.nn.Parameter(theta.clone().real)
        self.logits=torch.nn.Parameter(weights.clamp_min(1e-12).log())
        if diagonal:
            self.energy_raw=torch.nn.Parameter(torch.atanh(energy.diag().real.clamp(-.98,.98)))
        else:
            ee,V=torch.linalg.eigh((energy+energy.mH)/2)
            dd=(1+ee.clamp(-.98,.98))/(1-ee.clamp(-.98,.98))
            R=dd.sqrt()[:,None]*V.mH
            self.energy_raw=torch.nn.Parameter(torch.view_as_real(R).contiguous())
        self.register_buffer('q',torch.tensor(layout.q,dtype=RDTYPE,device=layout.device))
        keys=layout.keys
        self.register_buffer('lag',torch.tensor([k[1] for k in keys[:layout.ns]],dtype=RDTYPE,device=layout.device))
        self.register_buffer('s_imag',torch.tensor([k[-1]=='I' for k in keys[:layout.ns]],device=layout.device))
        self.register_buffer('hi',torch.tensor([k[1] for k in keys[layout.ns:]],device=layout.device))
        self.register_buffer('hj',torch.tensor([k[2] for k in keys[layout.ns:]],device=layout.device))
        self.register_buffer('h_imag',torch.tensor([k[-1]=='I' for k in keys[layout.ns:]],device=layout.device))
    def core(self):
        w=self.logits.softmax(0)
        V=w.sqrt()[:,None]*torch.exp(-1j*self.theta[:,None]*self.q[None,:])
        if self.diagonal:A=torch.diag(torch.tanh(self.energy_raw).to(CDTYPE))
        else:
            R=torch.view_as_complex(self.energy_raw)
            I=torch.eye(self.rank,device=R.device,dtype=CDTYPE)
            A=I-2*torch.linalg.solve(I+R.mH@R,I)
            A=(A+A.mH)/2
        return w,V,A
    def forward(self):
        w,V,A=self.core()
        ss=(w[:,None]*torch.exp(-1j*self.theta[:,None]*self.lag[None,:])).sum(0)
        hh=(V.mH@A@V)[self.hi,self.hj]
        return torch.cat((torch.where(self.s_imag,ss.imag,ss.real),
                          torch.where(self.h_imag,hh.imag,hh.real)))
    def green(self,info:dict,z:torch.Tensor):
        w,V,A=self.core()
        K=info['sign']*(info['scale']*A-info['shift']*torch.eye(self.rank,device=A.device,dtype=CDTYPE))
        return green(K,info['beta']*w.sqrt().to(CDTYPE),z)


def initialize_latent(y:torch.Tensor,layout:SettingLayout,info:dict,rank:int,
                      diagonal:bool) -> LatentPhysicalModel:
    S,T=layout.matrices(y,info)
    Hn=(info['sign']*T+info['shift']*S)/info['scale']
    order=sorted(range(layout.m),key=lambda j:layout.q[j])
    qs=[layout.q[i] for i in order]
    check(all(qs[k+1]-qs[k]==1 for k in range(len(qs)-1)),
          'This initializer requires consecutive signed time indices.')
    ix=torch.tensor(order,device=layout.device)
    ev,Q=torch.linalg.eigh(S[ix[:,None],ix[None,:]])
    Q=Q[:,-rank:]
    shift=torch.linalg.pinv(Q[:-1],rtol=1e-10)@Q[1:]
    theta=torch.angle(torch.linalg.eigvals(shift))
    q=torch.tensor(layout.q,dtype=RDTYPE,device=layout.device)
    D=torch.exp(-1j*theta[:,None]*q[None,:]);inv=torch.linalg.pinv(D,rtol=1e-8)
    weights=(inv.mH@S@inv).diag().real.clamp_min(.001);weights/=weights.sum()
    Z=weights.sqrt()[:,None]*D;inv=torch.linalg.pinv(Z,rtol=1e-8)
    A=inv.mH@Hn@inv;A=(A+A.mH)/2
    return LatentPhysicalModel(layout,rank,theta,weights,A,diagonal)


def train_latent(model:LatentPhysicalModel,y:torch.Tensor,counts:torch.Tensor,
                 max_steps:int) -> dict:
    """Finite-iteration local likelihood optimization; not a global certificate."""
    calls=0;scale=1e4
    with torch.enable_grad():
        opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=max_steps,
            line_search_fn='strong_wolfe',history_size=30,tolerance_grad=1e-7,
            tolerance_change=1e-10)
        def closure():
            nonlocal calls
            calls+=1;opt.zero_grad()
            objective=scale*binomial_deviance(model(),y,counts)
            check(bool(torch.isfinite(objective)),'Nonfinite likelihood.')
            objective.backward();return objective
        start=float(binomial_deviance(model(),y,counts).detach())
        opt.step(closure)
        end=float(binomial_deviance(model(),y,counts).detach())
    first=next(iter(opt.state.values()))
    gradient=max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None)
    check(end<=start+1e-9,'Optimizer increased likelihood loss.')
    with torch.no_grad():
        w,V,A=model.core()
        check(bool(torch.isfinite(model()).all()),'Nonfinite predictions.')
        check(float(model().abs().max())<=1+1e-8,'Latent outcome outside [-1,1].')
        check(float(torch.linalg.eigvalsh(A).abs().max())<=1+1e-9,'Latent energy bound violated.')
    return {'initial_deviance':start,'final_deviance':end,'closure_calls':calls,
            'iterations':int(first.get('n_iter',0)),'max_steps':max_steps,
            'iteration_cap_reached':int(first.get('n_iter',0))>=max_steps,
            'scaled_gradient_max':gradient,'global_optimum_certified':False}


def latent_cv(yt,yv,ct,cv,layout,info,z,diagonal,max_steps=FIT_STEPS,ranks=RANKS):
    models=[];audits=[]
    for rank in ranks:
        with torch.no_grad():model=initialize_latent(yt,layout,info,rank,diagonal)
        audit=train_latent(model,yt,ct,max_steps)
        with torch.no_grad():validation=float(binomial_deviance(model(),yv,cv))
        audits.append({'rank':rank,'validation_deviance':validation,'training':audit})
        models.append(model)
    winner=min(range(len(ranks)),key=lambda k:audits[k]['validation_deviance'])
    model=models[winner]
    combined=(ct*yt+cv*yv)/(ct+cv)
    refit=train_latent(model,combined,ct+cv,max_steps)
    with torch.no_grad():
        G=model.green(info,z).detach()
        w,V,A=model.core()
        residual=float(torch.linalg.vector_norm(A@torch.diag(torch.exp(-1j*model.theta))-
                                                torch.diag(torch.exp(-1j*model.theta))@A))
        snapshot={'phases':model.theta.detach().tolist(),'weights':w.tolist(),
                  'energy_real':A.real.tolist(),'energy_imag':A.imag.tolist()}
    return G,{'selected_rank':ranks[winner],'rank_trials':audits,'refit':refit,
              'phase_energy_commutator_norm':residual,'model':snapshot}


def new_tests() -> dict:
    """Algebra tests do not establish statistical or scientific superiority."""
    torch.set_num_threads(2);cpu=torch.device('cpu')
    layout=SettingLayout(time_indices6(12),cpu);rank=4
    theta=torch.tensor([-.9,-.2,.5,1.2],dtype=RDTYPE)
    weights=torch.tensor([.1,.2,.3,.4],dtype=RDTYPE)
    gen=torch.Generator().manual_seed(711)
    A=torch.randn((rank,rank),generator=gen,dtype=CDTYPE);A=(A+A.mH)/2
    A=A/(2*torch.linalg.matrix_norm(A,ord=2))
    model=LatentPhysicalModel(layout,rank,theta,weights,A,False)
    info={'scale':2.,'shift':-.7,'sign':1,'beta':math.sqrt(.5)}
    z=torch.linspace(-4,4,101,dtype=RDTYPE)+.2j
    with torch.no_grad():
        w,V,AA=model.core();y=model();S,T=layout.matrices(y,info)
        algebra=max(float((S-V.mH@V).abs().max()),float((AA-A).abs().max()))
        check(algebra<1e-10,'Latent forward algebra mismatch.')
        svdU,ss,vh=torch.linalg.svd(V,full_matrices=False)
        C=vh.mH/ss[None,:]
        G1=green(C.mH@T@C,C.mH@(info['beta']*S[:,0]),z)
        G2=model.green(info,z)
        agree=float((G1-G2).abs().max());check(agree<1e-8,'Latent/projected Green mismatch.')
        # Convex projection of exactly feasible measurements is a fixed point.
        nn=torch.full((layout.nsettings,),100000,dtype=torch.int64)
        fitted,audit=cone_fit(y,nn,layout,joint=True)
        fp=float((fitted-y).abs().max());check(fp<1e-7,'Cone fit changed feasible data.')
        mass=float(info['beta']**2*w.sum());check(abs(mass-.5)<1e-12,'Latent weight lost.')
        check(float((-G2.imag/math.pi).min())>=-1e-12,'Latent spectrum negative.')
    # Nonzero finite-difference check for likelihood gradient, full complex A.
    counts=torch.full((layout.nsettings,),10000,dtype=torch.int64)
    obs=(model().detach()+.002).clamp(-.99,.99)
    objective=binomial_deviance(model(),obs,counts);objective.backward()
    analytic=float(model.theta.grad[1]);h=1e-6
    with torch.no_grad():
        model.theta[1]+=h;plus=float(binomial_deviance(model(),obs,counts))
        model.theta[1]-=2*h;minus=float(binomial_deviance(model(),obs,counts))
        model.theta[1]+=h
    fd=(plus-minus)/(2*h);ge=abs(analytic-fd)
    check(ge<1e-7,'Likelihood gradient mismatch.')
    # A small Gram direction can carry O(1) spectral information, despite
    # PSD, bounded physical H, normalized positive spectra. Not a novelty claim.
    eps=1e-8;c=math.sqrt(1-eps)
    Phi=torch.tensor([[1.,c],[0.,math.sqrt(eps)]],dtype=CDTYPE)
    H0=torch.zeros((2,2),dtype=CDTYPE);H1=torch.tensor([[0.,.7],[.7,0.]],dtype=CDTYPE)
    source=torch.tensor([1.,0.],dtype=CDTYPE)
    delta=float((green(H0,source,z)-green(H1,source,z)).abs().max())
    data_delta=float(torch.linalg.vector_norm(Phi.mH@(H1-H0)@Phi))
    return {'latent_forward_max_error':algebra,'latent_vs_projected_max_error':agree,
            'convex_exact_fixed_point_error':fp,'likelihood_gradient_abs_error':ge,
            'positivity_and_mass':True,'no_go_small_input_norm':data_delta,
            'no_go_Green_max_difference':delta,
            'note':'Physical constraints alone do not ensure spectral accuracy.'}


def time_indices6(n:int)->list[int]:
    return [0]+[((j+1)//2)*(1 if j%2 else -1) for j in range(1,n)]


def run6(args,directory,log):
    global PRODUCT_STEP
    PRODUCT_STEP=args.product_step
    torch.set_num_threads(args.threads);device=get_device(args.device)
    dump_json(directory/'environment.json',environment(device))
    log(f'Cihaz: {device}; matris modelinin egitimi: cpu; PyTorch {torch.__version__}')
    log('[1] Fizik, model, olasilik, kisit ve turev testleri...')
    with torch.no_grad():tests=self_tests(device);tests.update(extra_tests(device))
    tests.update(new_tests());dump_json(directory/'self_tests.json',tests);log('Dogrulama: PASS')
    if args.suite=='smoke':cases=[4.];budgets=[10**8];n=2;start_seed=6000
    elif args.suite=='development':cases=[2.,4.,8.];budgets=[10**8];n=5;start_seed=6100
    else:cases=[5.,7.];budgets=[10**7,10**8,10**9];n=20;start_seed=7000
    if args.seeds is not None:n=args.seeds
    cfg={'version':VERSION,'suite':args.suite,'L':6,'U':cases,'budgets':budgets,
         'seeds':list(range(start_seed,start_seed+n)),'ranks':RANKS,'fit_steps':FIT_STEPS,
         'training_fraction':FIT_FRACTION,'input_circuits_per_branch':24,'eta':ETA,
         'spacing':SPACING,'product_step':PRODUCT_STEP,'layout':'prefix24','allocation':'uniform',
         'limits':['All points are exploratory once inspected; no automatic held-out claim.',
                   'One solver initialization; local optimizer; finite iteration cap.',
                   'No fitted spectral widths; fixed eta=0.2; no target spectrum in estimator.',
                   'Shared counted observations for all estimators, including rank selection.',
                   'Exact source and E0; ideal gates; no total physical runtime comparison.',
                   'PSD baselines are controls, not faithful reproductions of every published method.',
                   'No novelty, confidence coverage, global minimum or quantum advantage claim.']}
    cfg['settings_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    dump_json(directory/'config.json',cfg);log('Ayar SHA256: '+cfg['settings_sha256'])
    cpu=torch.device('cpu');layout=SettingLayout(time_indices6(24),cpu)
    omega=torch.linspace(-12,12,2401,dtype=RDTYPE);z=omega+1j*ETA
    variants=('raw_cutoff','gram_psd_cutoff','joint_psd_cutoff','latent_diagonal_cv','latent_joint_cv')
    rows=[];details={};summary=[];clock=time.perf_counter()
    for U in cases:
        log(f'[2] L=6 U/t={U:g}; ayni 24 devre, ayni toplam olcum')
        with torch.no_grad():
            prob=build_problem(6,U,device);backends=[]
            truth=-sum(green(br['K'],br['b'],z.to(device)) for br in prob['branches']).imag.cpu()/math.pi
            mass=float(integral(truth,omega))
            for bid,br in enumerate(prob['branches']):
                data,_,_=fixed_time_pool(6,U,br,24,SPACING,PRODUCT_STEP)
                backends.append(SamplingBackend(data,br,prob,layout,bid,'prefix24'))
        for B in budgets:
            cn=torch.full((layout.nsettings,),B//(2*layout.nsettings),dtype=torch.int64)
            ct=torch.floor(cn.to(RDTYPE)*FIT_FRACTION).to(torch.int64);cv=cn-ct
            check(int(cv.min())>0,'Budget too small for validation split.')
            used=2*int(cn.sum());check(used<=B,'Budget exceeded.')
            for seed in cfg['seeds']:
                greens={v:[] for v in variants};audits={v:[] for v in variants}
                t0=time.perf_counter()
                for bid,backend in enumerate(backends):
                    yt=backend.sample(ct,seed,'mavi_train');yv=backend.sample(cv,seed,'mavi_validation')
                    y=(ct*yt+cv*yv)/cn
                    dump_json(directory/f'raw_U{U:g}_B{B}_seed{seed}_branch{bid}.json',
                        {'keys':layout.keys,'q':layout.q,'info':backend.info,'counts_train':ct.tolist(),
                         'counts_validation':cv.tolist(),'observed_train':yt.tolist(),'observed_validation':yv.tolist()})
                    with torch.no_grad():
                        for kind in ('raw_cutoff','gram_psd_cutoff','joint_psd_cutoff'):
                            fitted,fitinfo=(y,{}) if kind=='raw_cutoff' else cone_fit(y,cn,layout,kind=='joint_psd_cutoff')
                            S,T=layout.matrices(fitted,backend.info)
                            G,si=project_measured(S,T,backend.info['beta'],z,int(cn[0]))
                            greens[kind].append(G);audits[kind].append({'fit':fitinfo,'solve':si})
                    for diagonal,name in ((True,'latent_diagonal_cv'),(False,'latent_joint_cv')):
                        G,audit=latent_cv(yt,yv,ct,cv,layout,backend.info,z,diagonal)
                        greens[name].append(G);audits[name].append(audit)
                key=f'U{U:g}_B{B}_seed{seed}';details[key]=audits
                curves={'omega':omega.tolist(),'exact':truth.tolist()};values={}
                for name in variants:
                    spec=-sum(greens[name]).imag/math.pi
                    check(bool(torch.isfinite(spec).all()) and float(spec.min())>=-1e-8,'Invalid spectrum.')
                    err=float(integral((spec-truth).abs(),omega))/mass;values[name]=err
                    row={'U':U,'budget':B,'seed':seed,'method':name,'relative_L1_window':err,
                         'actual_shots':used,'unused_shots':B-used,
                         'shot_weighted_rotation_proxy':2*int((cn*layout.time_tick_cost).sum())*round(SPACING/PRODUCT_STEP)*(4*(6-1)+6)}
                    rows.append(row);curves[name]=spec.tolist()
                csv_write(directory/f'curves_{key}.csv',[{k:v[j] for k,v in curves.items()} for j in range(len(omega))])
                csv_write(directory/'metrics.csv',rows);dump_json(directory/'details.json',details)
                log(f'  B={B:.0e}, seed={seed}: '+ '; '.join(f'{k}=%{100*v:.3f}' for k,v in values.items()))
            for name in variants:
                group=[r for r in rows if r['U']==U and r['budget']==B and r['method']==name]
                summary.append({'U':U,'budget':B,'method':name,'n':len(group),
                    'median_error':statistics.median(r['relative_L1_window'] for r in group),
                    'min_error':min(r['relative_L1_window'] for r in group),
                    'max_error':max(r['relative_L1_window'] for r in group)})
    dump_json(directory/'summary.json',{'status':'PASS','config':cfg,'summary':summary,
        'seconds':time.perf_counter()-clock,'self_tests':tests})
    log('\nSONUC: PASS (calisma/dogruluk kontrolleri; yeni yontem basarisi degil)')
    for U in cases:
        for B in budgets:
            group={r['method']:r for r in summary if r['U']==U and r['budget']==B}
            log(f'U/t={U:g}, B={B:.0e}, n={n}: '+'; '.join(f'{k}=%{100*group[k]["median_error"]:.3f}' for k in variants))
    log('Tum yontemler ayni olcum verisini kullanir. Rank secimi gercek spektrumu gormez.')
    log('Fiziksel/pozitif sonuc dogru sonuc garantisi degildir; optimizasyon yereldir.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--product-step',type=float,choices=(.125,.25),default=.125)
    parser.add_argument('--suite',choices=('smoke','development','validation'),default='development')
    parser.add_argument('--out',type=Path,default=Path.home()/'spektrum_deney6')
    parser.add_argument('--seeds',type=int,default=None)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.threads<=32:parser.error('--threads 1..32 olmali')
    if args.seeds is not None and not 1<=args.seeds<=100:parser.error('--seeds 1..100 olmali')
    root=args.out.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    directory=root/f'run6_{tag}';directory.mkdir()
    def log(s):
        print(s,flush=True)
        with (directory/'run.log').open('a',encoding='utf-8') as f:f.write(s+'\n')
    status=0
    try:
        source=Path(__file__).read_bytes();(directory/'script_snapshot.py').write_bytes(source)
        dump_json(directory/'provenance.json',{'sha256':hashlib.sha256(source).hexdigest(),'argv':sys.argv,'version':VERSION})
        run6(args,directory,log)
    except (Exception,KeyboardInterrupt) as exc:
        status=1;text=traceback.format_exc();(directory/'error.txt').write_text(text)
        log('\nHATA: '+str(exc));log(text)
    finally:
        archive=root/f'rapor6_{tag}.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(directory.rglob('*')):
                if p.is_file():zf.write(p,arcname=p.relative_to(directory))
        print('\nRAPOR_ZIP: '+str(archive),flush=True)
    return status
if __name__=='__main__':raise SystemExit(main())
