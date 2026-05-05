"""
PPO Training v5 — LFP Observability Fix
==========================================
Builds on v4 (chemistry + pack-size generalization) with three targeted
improvements to close the LFP reliability gap:

1. State expansion: 9D → 11D
   obs[9]  = soc_cc: Coulomb-counted SOC (OCV-independent running integral)
   obs[10] = dsoc_dt: SOC velocity (× 100 for scale) — tells the policy
             how fast charge is accumulating regardless of OCV shape.

2. LFP oversampling: 50% vs 33%
   Root-cause analysis of v4 failures: LFP episodes that fail use only
   ~62% of max current (vs ~95% for successful episodes), indicating the
   policy is too conservative. More LFP gradient updates fix this faster.

3. Tighter LFP ambient temperature: ±2°C vs ±5°C
   LFP has fewer steps of slack (1080 ideal vs 2400 limit). Reducing
   thermal variance in training stabilises the current-maintenance policy.

Warm-start: if ppo_outputs_v4/ppo_best.pt exists, backbone weights for
the 9 shared state dimensions are copied; the two new columns (soc_cc,
dsoc_dt) are initialised randomly. NMC/LCO performance is preserved
from day 1; LFP benefits from all three fixes simultaneously.

Previous version (v4):
  Chemistry + Pack-Size Generalization
=============================================================
Extends v3 by treating pack size (n_cells) as a second contextual
variable alongside chemistry (θ), enabling a single policy to
generalize across:

  Chemistry θ  ∈  {NMC, LCO, LFP}
  Pack layout  ∈  {2×2, 2×3, 3×3, 3×4, 4×4}   (4–16 cells)

Key architectural change from v3 (pure chemistry generalization):
  - obs_dim: 8 → 9  (+n_cells_norm as 9th element)
  - Action: I_pack → I_cell  (per-cell current; I_pack = I_cell × n_cells)
  - Physics steps are IDENTICAL for all pack sizes at fixed I_cell
  - Thermal coupling topology differs → encoded implicitly via n_cells_norm
    and explicitly via sigma_T which captures pack-level gradients

Why I_cell generalization works:
  The surrogate models predict ΔT and ΔSOC for a SINGLE CELL given
  I_cell. The pack simulator then applies cell-to-cell conduction.
  With I_cell as the action, the policy learns "apply 5A/cell" and
  the simulator handles I_pack = I_cell × n_cells internally.
  Episode length (steps to SOC=0.80) depends only on I_cell and C_nom,
  not on n_cells — so training signal is equally strong across all sizes.

Curriculum strategy:
  No explicit curriculum needed. The 3×3 layout is in the training mix,
  preserving learned behavior from v3. Smaller packs (2×2, 2×3) are
  easier (less thermal coupling variance) and converge early; larger
  packs (3×4, 4×4) take slightly longer due to higher sigma_T.
"""

import os, sys, time, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from Scripts.charging_env_v4 import BatteryPackEnv, PACK_LAYOUTS, I_CELL_MAX, I_CELL_MIN

OUTPUT_DIR = r'ppo_outputs_v5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────
LR          = 3e-4    # slightly reduced for 9D obs (more stable early)
GAMMA       = 0.999
LAM         = 0.95
CLIP_EPS    = 0.20
ENT_COEF    = 0.002
ENT_TARGET  = 0.50    # entropy ceiling (nats); same fix as v3
VF_COEF     = 0.50
MAX_GRAD    = 0.50
N_STEPS     = 2048
N_EPOCHS    = 4
MINI_BATCH  = 128
TOTAL_STEPS = 8_000_000   # continue from v4 checkpoint if available
EVAL_EVERY  = 100_000

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ── Physics-compatible limits ─────────────────────────────────────
I_CELL_MAX_PPO = I_CELL_MAX     # 5.0 A/cell
I_CELL_MIN_PPO = I_CELL_MIN     # 0.05 A/cell
MDOT_FIXED     = 0.02           # kg/s fixed cooling
MAX_STEPS_ENV  = 2400           # timestep limit per episode


# ══════════════════════════════════════════════════════════════════
# ENVIRONMENT WRAPPER  (1D action: I_cell only, fixed mdot)
# ══════════════════════════════════════════════════════════════════

class ChargingEnvV4:
    """
    Single-action wrapper (I_cell ∈ [-1,1]).
    Fixed mdot = MDOT_FIXED for simplicity during initial training.
    Pack size is randomized by the underlying BatteryPackEnv each episode.
    """

    def __init__(self, base_env: BatteryPackEnv):
        self.env          = base_env
        self.obs_dim      = base_env.obs_dim   # 9
        self.act_dim      = 1
        self._available   = base_env._available
        self._prev_soc_min = SOC_INIT = 0.2
        self._steps        = 0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._prev_soc_min = float(obs[1])   # SOC_min
        self._steps = 0
        return obs, info

    def step(self, action):
        a      = float(action[0])
        I_cell = (a + 1.0) / 2.0 * (I_CELL_MAX_PPO - I_CELL_MIN_PPO) + I_CELL_MIN_PPO
        mdot_n = float((MDOT_FIXED / 0.05) * 2 - 1)   # fixed mdot in [-1,1]
        base_action = np.array([a, mdot_n], dtype=np.float32)

        obs, _, done, trunc, info = self.env.step(base_action)
        self._steps += 1

        reason       = info.get('done_reason', '')
        curr_soc_min = float(obs[1])

        d_soc  = max(0.0, curr_soc_min - self._prev_soc_min)
        reward = 50.0 * d_soc - 0.01

        if reason == 'success':
            reward += 100.0
        elif reason in ('thermal_abort', 'safety_abort'):
            reward -= 30.0

        self._prev_soc_min = curr_soc_min

        if self._steps >= MAX_STEPS_ENV and not done:
            done  = True; trunc = False
            info['done_reason'] = 'timeout'

        return obs, reward, done, trunc, info

    @property
    def sim(self):          return self.env.sim
    @property
    def step_count(self):   return self._steps
    @property
    def chemistry(self):    return self.env.chemistry
    @property
    def layout(self):       return self.env.layout
    @property
    def fixed_chemistry(self): return self.env.fixed_chemistry
    @fixed_chemistry.setter
    def fixed_chemistry(self, v): self.env.fixed_chemistry = v
    @property
    def fixed_layout(self): return self.env.fixed_layout
    @fixed_layout.setter
    def fixed_layout(self, v): self.env.fixed_layout = v


# ══════════════════════════════════════════════════════════════════
# RUNNING MEAN/STD NORMALISER
# ══════════════════════════════════════════════════════════════════

class RunningMeanStd:
    """Welford online algorithm for observation normalisation."""
    def __init__(self, shape):
        self.mean  = np.zeros(shape, np.float64)
        self.var   = np.ones(shape,  np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        if x.ndim == 1: x = x[np.newaxis]
        bm = x.mean(0); bv = x.var(0); bn = len(x)
        d  = bm - self.mean
        tot = self.count + bn
        self.mean += d * bn / tot
        self.var   = (self.var*self.count + bv*bn + d**2*self.count*bn/tot) / tot
        self.count = tot

    def normalise(self, x):
        return ((np.asarray(x, np.float32) - self.mean.astype(np.float32))
                / (np.sqrt(self.var.astype(np.float32)) + 1e-8)).clip(-10, 10)


# ══════════════════════════════════════════════════════════════════
# ACTOR-CRITIC  (identical architecture to v3; obs_dim=9)
# ══════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=11, act_dim=1, hidden=256):  # 11 = 9 + soc_cc + dsoc_dt
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        # log_std clamped in forward to [-3, -0.9] → entropy ceiling ~0.51 nats
        self.log_std    = nn.Parameter(-2.0 * torch.ones(act_dim))
        self.critic     = nn.Linear(hidden, 1)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 3.0)  # tanh(3)≈0.995 → I_cell≈4.97A
        nn.init.orthogonal_(self.critic.weight,    gain=1.0)

    def forward(self, obs):
        x        = self.backbone(obs)
        mean     = torch.tanh(self.actor_mean(x))
        log_std  = self.log_std.clamp(-3.0, -0.9)   # hard entropy ceiling
        value    = self.critic(x).squeeze(-1)
        return mean, log_std.expand_as(mean), value

    def get_action(self, obs, deterministic=False):
        mean, log_std, value = self.forward(obs)
        std  = log_std.exp().clamp(0.01, 1.0)
        dist = torch.distributions.Normal(mean, std)
        a    = mean if deterministic else dist.sample()
        a    = a.clamp(-1.0, 1.0)
        lp   = dist.log_prob(a).sum(-1)
        return a, lp, value

    def evaluate(self, obs, action):
        mean, log_std, value = self.forward(obs)
        std  = log_std.exp().clamp(0.01, 1.0)
        dist = torch.distributions.Normal(mean, std)
        lp   = dist.log_prob(action).sum(-1)
        ent  = dist.entropy().sum(-1)
        return lp, ent, value


# ══════════════════════════════════════════════════════════════════
# ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self, n, obs_dim, act_dim):
        self.n    = n
        self.obs  = np.zeros((n, obs_dim),  np.float32)
        self.acts = np.zeros((n, act_dim),  np.float32)
        self.rews = np.zeros(n,             np.float32)
        self.vals = np.zeros(n,             np.float32)
        self.lps  = np.zeros(n,             np.float32)
        self.done = np.zeros(n,             np.float32)
        self.adv  = np.zeros(n,             np.float32)
        self.ret  = np.zeros(n,             np.float32)
        self.ptr  = 0

    def add(self, obs, act, rew, val, lp, done):
        i = self.ptr
        self.obs[i]=obs; self.acts[i]=act; self.rews[i]=rew
        self.vals[i]=val; self.lps[i]=lp; self.done[i]=done
        self.ptr += 1

    def compute_gae(self, last_val):
        gae = 0.0
        for t in reversed(range(self.n)):
            nv = last_val if t == self.n-1 else self.vals[t+1]
            nd = 0.0      if t == self.n-1 else self.done[t+1]
            delta = self.rews[t] + GAMMA*nv*(1-nd) - self.vals[t]
            gae   = delta + GAMMA*LAM*(1-self.done[t])*gae
            self.adv[t] = gae
        self.ret  = self.adv + self.vals
        self.adv  = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)
        self.ptr  = 0

    def batches(self, bs):
        idx = np.random.permutation(self.n)
        for s in range(0, self.n, bs):
            b = idx[s:s+bs]
            yield (torch.FloatTensor(self.obs[b]),
                   torch.FloatTensor(self.acts[b]),
                   torch.FloatTensor(self.lps[b]),
                   torch.FloatTensor(self.adv[b]),
                   torch.FloatTensor(self.ret[b]))


# ══════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate(policy, env, obs_rms, device, n_ep=2):
    """Evaluate all (chemistry × layout) combinations."""
    results = {}
    for chem in env._available:
        for layout in PACK_LAYOUTS:
            env.fixed_chemistry = chem
            env.fixed_layout    = layout
            key = f'{chem}_{layout[0]}x{layout[1]}'
            chem_res = []
            for _ in range(n_ep):
                obs, _ = env.reset()
                total_r = 0; done = trunc = False
                while not (done or trunc):
                    obs_n = obs_rms.normalise(obs)
                    ot    = torch.FloatTensor(obs_n).unsqueeze(0).to(device)
                    with torch.no_grad():
                        a, _, _ = policy.get_action(ot, deterministic=True)
                    obs, r, done, trunc, info = env.step(a.cpu().numpy()[0])
                    total_r += r
                sm = env.sim.get_pack_summary()
                chem_res.append({
                    'reward': total_r, 'steps': env.step_count,
                    'SOC': sm['SOC_min'], 'T_max': sm['T_max'],
                    'reason': info.get('done_reason', '?'),
                    'n_cells': sm['n_cells'],
                })
            results[key] = chem_res
    env.fixed_chemistry = None
    env.fixed_layout    = None
    return results


def print_eval(results, step):
    print(f'\n  ── Eval @ step {step:,} ──')
    # Group by chemistry for compact display
    for chem in ['NMC', 'LFP', 'LCO']:
        for layout in PACK_LAYOUTS:
            key = f'{chem}_{layout[0]}x{layout[1]}'
            if key not in results: continue
            eps = results[key]
            r   = np.mean([e['reward'] for e in eps])
            ok  = sum(1 for e in eps if e['reason']=='success')
            n   = eps[0]['n_cells']
            st  = np.mean([e['steps'] for e in eps])
            T   = np.mean([e['T_max'] for e in eps])
            print(f'    {chem} {layout[0]}x{layout[1]} ({n:2d}cells): '
                  f'r={r:7.2f}  ok={ok}/{len(eps)}  '
                  f'steps={st:.0f}  T={T:.1f}°C')


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    print('Loading environments...')
    env      = ChargingEnvV4(BatteryPackEnv(T_ambient=25.0))
    eval_env = ChargingEnvV4(BatteryPackEnv(T_ambient=25.0))
    print()

    # Physics sanity check (per-cell physics is layout-independent)
    print('Physics check (steps at I_cell=5A, layout-independent):')
    for chem, cn in [('LFP',2.5),('NMC',3.0),('LCO',1.1)]:
        s = int(0.6 * cn * 3600 / I_CELL_MAX_PPO)
        ok = '✓' if s < MAX_STEPS_ENV else '✗ INCREASE MAX_STEPS_ENV'
        print(f'  {chem}: {s} steps  {ok}')
    print(f'  All layouts share the same per-cell step count.')
    print(f'  n_cells_norm ∈ '
          f'[{4/16:.2f}, {16/16:.2f}] across {len(PACK_LAYOUTS)} layouts')
    print()

    policy  = ActorCritic(env.obs_dim, env.act_dim).to(device)
    opt     = Adam(policy.parameters(), lr=LR, eps=1e-5)
    buf     = RolloutBuffer(N_STEPS, env.obs_dim, env.act_dim)
    obs_rms = RunningMeanStd(env.obs_dim)

    # Optional warm-start: load backbone weights from v4 checkpoint.
    # The v4 policy has obs_dim=9; we load only the backbone + actor_mean
    # layers (which are compatible) and leave soc_cc/dsoc_dt weights at
    # their orthogonal random init.  This gives NMC/LCO a head-start while
    # LFP learns its new state elements from scratch.
    V4_CKPT = os.path.join('ppo_outputs_v4', 'ppo_best.pt')
    if os.path.exists(V4_CKPT):
        try:
            v4_sd = torch.load(V4_CKPT, map_location='cpu')
            v5_sd = policy.state_dict()
            # backbone weight shapes: v4 Linear(9,256) vs v5 Linear(11,256)
            # — only layer that differs is backbone.0.weight (first layer)
            # All other layers (backbone.2, actor_mean, log_std, critic) match exactly
            for k in v5_sd:
                if k == 'backbone.0.weight':
                    # v4: (256,9), v5: (256,11)
                    # Copy the 9 shared columns; new cols (soc_cc, dsoc_dt) keep init
                    v5_sd[k][:, :9] = v4_sd[k]
                elif k == 'backbone.0.bias':
                    v5_sd[k] = v4_sd[k]
                elif k in v4_sd and v4_sd[k].shape == v5_sd[k].shape:
                    v5_sd[k] = v4_sd[k]
            policy.load_state_dict(v5_sd)
            print(f'  [OK] Warm-started from {V4_CKPT} '
                  f'(backbone cols 0:9 copied; soc_cc/dsoc_dt cols random-init)')
        except Exception as e:
            print(f'  [WARN] Warm-start failed: {e} — training from scratch')
    else:
        print(f'  [INFO] No v4 checkpoint found — training from scratch')

    # Verify initial action
    test_obs = torch.zeros(1, env.obs_dim).to(device)
    with torch.no_grad():
        a0, _, _ = policy.get_action(test_obs)
    I_cell_init = (a0.item()+1)/2*(I_CELL_MAX_PPO-I_CELL_MIN_PPO)+I_CELL_MIN_PPO
    print(f'Init policy: action={a0.item():.3f} → I_cell={I_cell_init:.2f}A')
    print(f'  For 9-cell pack: I_pack={I_cell_init*9:.1f}A')
    print(f'  For 4-cell pack: I_pack={I_cell_init*4:.1f}A')
    print(f'Policy params: {sum(p.numel() for p in policy.parameters()):,}  (obs_dim=11)')
    print(f'Context space: {len(["NMC","LFP","LCO"])*len(PACK_LAYOUTS)} combos '
          f'({len(["NMC","LFP","LCO"])} chem × {len(PACK_LAYOUTS)} layouts)')
    print()

    history     = {'ep_r':[], 'ep_st':[], 'chem':[], 'layout':[], 'eval':{}}
    best_eval_r = -np.inf
    obs, _      = env.reset()
    ep_r = 0; ep_st = 0; ep_n = 0
    total_steps = 0; last_eval = 0
    t0 = time.time()

    while total_steps < TOTAL_STEPS:

        # ── Collect rollout ──────────────────────────────────────
        for _ in range(N_STEPS):
            obs_rms.update(obs)
            obs_norm = obs_rms.normalise(obs)
            ot = torch.FloatTensor(obs_norm).unsqueeze(0).to(device)
            with torch.no_grad():
                a, lp, v = policy.get_action(ot)
            a_np = a.cpu().numpy()[0]

            obs2, r, done, trunc, info = env.step(a_np)
            buf.add(obs_norm, a_np, r, v.cpu().item(),
                    lp.cpu().item(), float(done or trunc))

            obs=obs2; ep_r+=r; ep_st+=1; total_steps+=1

            if done or trunc:
                history['ep_r'].append(ep_r)
                history['ep_st'].append(ep_st)
                history['chem'].append(env.chemistry)
                history['layout'].append(env.layout)
                ep_n+=1; ep_r=0; ep_st=0
                obs, _ = env.reset()

        # Bootstrap
        obs_rms.update(obs)
        ot = torch.FloatTensor(obs_rms.normalise(obs)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, lv = policy.get_action(ot)
        buf.compute_gae(lv.cpu().item())

        # ── PPO update with entropy clipping ─────────────────────
        pgs=[]; vfs=[]; ents=[]
        for _ in range(N_EPOCHS):
            for ob,ac,lo,ad,re in buf.batches(MINI_BATCH):
                ob=ob.to(device); ac=ac.to(device)
                lo=lo.to(device); ad=ad.to(device); re=re.to(device)
                ln, ent, val = policy.evaluate(ob, ac)
                ratio  = (ln - lo).exp()
                pg     = -torch.min(ratio*ad,
                                    ratio.clamp(1-CLIP_EPS,1+CLIP_EPS)*ad).mean()
                vf     = ((val - re)**2).mean()
                # Entropy clipping: penalise only when ent > ENT_TARGET
                ent_loss = torch.clamp(ent.mean() - ENT_TARGET, min=0.0)
                loss   = pg + VF_COEF*vf + ENT_COEF*ent_loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
                opt.step()
                pgs.append(pg.item()); vfs.append(vf.item())
                ents.append(ent.mean().item())

        # ── Logging ──────────────────────────────────────────────
        if ep_n > 0:
            r20  = np.mean(history['ep_r'][-20:])
            st20 = np.mean(history['ep_st'][-20:])
            ok20 = sum(1 for r in history['ep_r'][-20:] if r > 20)
            el   = time.time() - t0
            print(f'  step={total_steps:7,} | ep={ep_n:5d} | '
                  f'r={r20:8.2f} | steps={st20:.0f} | '
                  f'ok/20={ok20} | '
                  f'pg={np.mean(pgs):.4f} | ent={np.mean(ents):.3f} | '
                  f'{el:.0f}s')
            t0 = time.time()

        # ── Evaluation ───────────────────────────────────────────
        if total_steps - last_eval >= EVAL_EVERY:
            ev = evaluate(policy, eval_env, obs_rms, device, n_ep=2)
            print_eval(ev, total_steps)
            history['eval'][total_steps] = ev
            avg_ev = np.mean([e['reward']
                              for eps in ev.values() for e in eps])
            if avg_ev > best_eval_r:
                best_eval_r = avg_ev
                torch.save(policy.state_dict(),
                           os.path.join(OUTPUT_DIR, 'ppo_best.pt'))
                with open(os.path.join(OUTPUT_DIR, 'obs_rms.pkl'), 'wb') as f:
                    pickle.dump(obs_rms, f)
                print(f'  *** Best r={best_eval_r:.2f} saved ***')
            last_eval = total_steps
            with open(os.path.join(OUTPUT_DIR, 'history.pkl'), 'wb') as f:
                pickle.dump(history, f)

    torch.save(policy.state_dict(),
               os.path.join(OUTPUT_DIR, 'ppo_final.pt'))
    with open(os.path.join(OUTPUT_DIR, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f'\nDone. Best eval r={best_eval_r:.2f}')
    return policy, history


if __name__ == '__main__':
    print('='*65)
    print('PPO v4 — Chemistry + Pack-Size Generalization')
    print('  θ ∈ {NMC, LCO, LFP}  ×  n_cells ∈ {4, 6, 9, 12, 16}')
    print('='*65)
    train()
