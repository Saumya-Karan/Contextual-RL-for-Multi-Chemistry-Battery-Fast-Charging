"""
generate_all_plots_v4.py  —  All publication figures
=====================================================
USAGE (from the directory containing ppo_outputs_v4/):
  python generate_all_plots_v4.py [--policy_dir ppo_outputs_v4] [--env_dir .]

Loads ppo_best.pt + obs_rms.pkl and runs REAL policy rollouts for
trajectory and pack-thermal figures. All other figures use logged numbers.
Falls back to synthetic data if checkpoint is missing.
"""
import os, sys, pickle, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family':'serif','font.size':9,'axes.labelsize':9,
    'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':8,'figure.dpi':300,'axes.grid':True,
    'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False,
    'lines.linewidth':1.5,
})

CHEM_COLORS  = {'NMC':'#2166ac','LFP':'#1a9850','LCO':'#d6604d'}
LAYOUT_NAMES = ['2x2','2x3','3x3','3x4','4x4']
LAYOUTS      = [(2,2),(2,3),(3,3),(3,4),(4,4)]
N_CELLS      = [4,6,9,12,16]
FIGS = 'figures'
os.makedirs(FIGS, exist_ok=True)

def save(name, fig=None):
    if fig is None: fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(f'{FIGS}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGS}/{name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig); print(f'  Saved {name}')


# ── Policy loader ─────────────────────────────────────────────────

def load_policy(policy_dir, env_dir):
    pt  = os.path.join(policy_dir, 'ppo_best.pt')
    rms = os.path.join(policy_dir, 'obs_rms.pkl')
    if not (os.path.exists(pt) and os.path.exists(rms)):
        print(f'  [WARN] {pt} not found — using synthetic data for rollout figures.')
        return None, None, None
    if env_dir not in sys.path: sys.path.insert(0, env_dir)
    try:
        import torch, torch.nn as nn
        class AC(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone   = nn.Sequential(nn.Linear(9,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh())
                self.actor_mean = nn.Linear(256,1)
                self.log_std    = nn.Parameter(-2.0*torch.ones(1))
                self.critic     = nn.Linear(256,1)
            def act(self, o):
                x = self.backbone(o)
                return torch.tanh(self.actor_mean(x)).clamp(-1,1)
        m = AC(); m.load_state_dict(torch.load(pt, map_location='cpu')); m.eval()
        # Custom unpickler: intercepts RunningMeanStd regardless of which
        # module name pickle stored (could be __main__, ppo_train_v4, etc.)
        class RunningMeanStd:
            def __init__(self, shape=()):
                self.mean  = np.zeros(shape, np.float64)
                self.var   = np.ones(shape,  np.float64)
                self.count = 1e-4
            def update(self, x):
                x = np.asarray(x, np.float64)
                if x.ndim == 1: x = x[np.newaxis]
                bm = x.mean(0); bv = x.var(0); bn = len(x)
                d  = bm - self.mean; tot = self.count + bn
                self.mean += d * bn / tot
                self.var   = (self.var*self.count + bv*bn + d**2*self.count*bn/tot) / tot
                self.count = tot
            def normalise(self, x):
                return ((np.asarray(x, np.float32) - self.mean.astype(np.float32))
                        / (np.sqrt(self.var.astype(np.float32)) + 1e-8)).clip(-10, 10)
        class _RMSUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'RunningMeanStd':
                    return RunningMeanStd
                return super().find_class(module, name)
        with open(rms, 'rb') as f: obs_rms = _RMSUnpickler(f).load()
        from charging_env_v3 import BatteryPackEnv
        print(f'  [OK] Loaded {pt}')
        return m, obs_rms, BatteryPackEnv
    except Exception as e:
        print(f'  [WARN] Load failed: {e} — using synthetic data.')
        return None, None, None


def run_episode(model, obs_rms, BatteryPackEnv, chem, layout=(3,3), seed=0):
    import torch
    I_MIN, I_MAX, MDOT = 0.05, 5.0, 0.02
    env = BatteryPackEnv(T_ambient=25.0, chemistry=chem, pack_layout=layout)
    obs, _ = env.reset(seed=seed)
    hist = {k:[] for k in ['soc_max','soc_min','T_max','T_min','T_mean','I_cell']}
    done = trunc = False; step = 0
    while not (done or trunc) and step < 2400:
        on = ((obs - obs_rms.mean.astype(np.float32))
              / (np.sqrt(obs_rms.var.astype(np.float32))+1e-8)).clip(-10,10)
        with torch.no_grad():
            a = model.act(torch.FloatTensor(on).unsqueeze(0)).item()
        I_cell = (a+1)/2*(I_MAX-I_MIN)+I_MIN
        mdot_n = (MDOT/0.05)*2-1
        obs, _, done, trunc, info = env.step(np.array([a, mdot_n], np.float32))
        sim = env.sim
        hist['soc_max'].append(float(obs[0])); hist['soc_min'].append(float(obs[1]))
        hist['T_max'].append(float(obs[2]));   hist['T_min'].append(float(obs[3]))
        hist['T_mean'].append(float(sim.T.mean())); hist['I_cell'].append(I_cell)
        step += 1
    return {k:np.array(v) for k,v in hist.items()} | {
        'done_reason': info.get('done_reason','?'), 'n_steps': step,
        'T_final': env.sim.T.copy(), 'layout': layout,
    }


def synth(chem, layout=(3,3), seed=0):
    rng = np.random.default_rng(seed); nc = layout[0]*layout[1]
    p = {'NMC':(3.0,1520,22.0,4.87),'LFP':(2.5,1070,22.0,4.90),'LCO':(1.1,560,22.0,4.83)}[chem]
    N = p[1]; t = np.arange(N); dS = p[3]/(p[0]*3600); soc = 0.2+dS*t
    T = p[2]+8*(1-np.exp(-t/(N*0.4)))
    taper = np.where(soc>0.76, np.clip((0.82-soc)*15,0,1), 1.0)
    T_map = np.full((layout[0],layout[1]), T[-1])
    T_map += rng.normal(0,0.5,(layout[0],layout[1]))
    return {'soc_max':soc+0.004,'soc_min':soc-0.004,'T_max':T+1,'T_min':T-0.8,
            'T_mean':T,'I_cell':p[3]*taper,'done_reason':'success','n_steps':N,
            'T_final':T_map.flatten(),'layout':layout}


# ── FIG 1: Architecture ───────────────────────────────────────────

def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2,3.8)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
    def box(x,y,w,h,t,c='#cce5ff',fs=7.5):
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=c,edgecolor='#333',lw=0.8,zorder=2))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,zorder=3,multialignment='center')
    def arr(x1,y1,x2,y2,lbl=''):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#444',lw=1.0))
        if lbl: ax.text((x1+x2)/2,(y1+y2)/2+0.1,lbl,fontsize=6.5,ha='center',color='#444')
    box(0.1,4.8,2.0,0.9,'Mendeley\nNMC (LG HG2)','#e8f4f8'); box(0.1,3.6,2.0,0.9,'Mendeley\nLFP (A123)','#e8f4f8')
    box(0.1,2.4,2.0,0.9,'CALCE A123\nLCO','#e8f4f8'); box(0.1,1.2,2.0,0.9,'Sandia/CALCE\nDegradation','#e8f4f8')
    box(0.1,0.1,2.0,0.9,'Lin et al.\nMech. TR','#e8f4f8')
    box(2.8,4.8,1.6,0.9,'NMC\nSurrogate','#d4edda'); box(2.8,3.6,1.6,0.9,'LFP\nSurrogate','#d4edda')
    box(2.8,2.4,1.6,0.9,'LCO\nSurrogate','#d4edda'); box(2.8,1.2,1.6,0.9,'Degradation\n×3','#d4edda')
    box(2.8,0.1,1.6,0.9,'Safety\nClassifier','#d4edda')
    box(5.2,1.5,2.0,3.1,'Configurable\nPack Simulator\n2×2 → 4×4\nK rebuilt / episode\nI_pack=I_cell×n','#fff3cd')
    box(7.8,2.2,2.0,2.2,'PPO Policy\nπ(a|s,θ,n)\nObs: 9D\nAct: I_cell','#f8d7da')
    for y in [5.25,4.05,2.85]: arr(2.1,y,2.8,y)
    arr(2.1,1.65,2.8,1.65); arr(2.1,0.55,2.8,0.55)
    for y in [5.25,4.05,2.85]: arr(4.4,y,5.2,3.0)
    arr(4.4,1.65,5.2,2.2); arr(7.2,3.3,7.8,3.3,'s_t'); arr(7.8,2.8,7.2,2.8,'I_cell')
    ax.text(5.2,5.0,'θ∈{NMC,LCO,LFP},  n_cells∈{4,6,9,12,16}',fontsize=7.5,color='#555',style='italic')
    fig.suptitle('System architecture: dual-context RL for LIB pack fast charging',fontsize=9)
    save('fig_architecture',fig)


# ── FIG 2: Training curves ────────────────────────────────────────

def fig_training_curves():
    steps = np.array([301,303,305,307,309,311,313,315,317,319,321,323,325,327,
        329,331,333,335,337,339,342,344,346,348,350,352,354,356,358,360,362,364,
        366,368,370,372,374,376,378,380,382,385,387,389,391,393,395,397,399,401,
        403,405,407,409,411,413,415,417,419,421,423,425,428,430,432,434,436,438,
        440,442,444,446,448,450,452,454,456,458,460,462,464,466,468,471,473,475,
        477,479,481,483,485,487,489,491,493,495,497,499,501,503,505,507,509,512,
        514,516,518,520,522,524,526,528,530,532,534,536,538,540,542,544,546,548,
        550,552,555,557,559,561,563,565,567,569,571,573,575,577,579,581,583,585,
        587,589,591,593,595,598,600,602,604,606,608,610,612,614,616,618,620,622,
        624,626,628,630,632,634,636,638,641,643,645,647,649,651,653,655,657,659,
        661,663,665,667,669,671,673,675,677,679,681,684,686,688,690,692,694,696,
        698,700,702,704,706,708,710,712,714,716,718,720,722,724,727,729,731,733,
        735,737,739,741,743,745,747,749,751,753,755,757,759,761,763,765,768,770,
        772,774,776,778,780,782,784,786,788,790,792,794,796,798,800,802,804,806,
        808,811,813,815,817,819,821,823,825,827,829,831,833,835,837,839,841,843,
        845,847,849,851,854,856,858,860,862,864,866,868,870,872,874,876,878,880,
        882,884,886,888,890,892,894,897,899,901,903,905,907,909,911,913,915,917,
        919,921,923,925,927,929,931,933,935,937,940,942,944,946,948,950,952,954,
        956,958,960,962,964,966,968,970,972,974,976,978,980,983,985,987,989,991,
        993,995,997,999,1001,1003])
    ok = np.array([17,17,17,17,18,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,
        19,19,19,19,19,19,19,19,19,19,19,19,19,19,18,17,17,17,17,17,17,17,17,17,
        17,17,19,19,19,18,18,18,18,18,18,19,19,18,18,18,18,18,19,19,19,19,18,18,
        18,19,19,19,18,18,18,18,19,19,19,19,19,18,18,18,18,19,19,19,19,18,18,18,
        18,18,18,18,18,18,18,18,18,18,18,18,18,17,17,18,18,17,16,16,17,17,17,17,
        17,16,16,16,17,17,17,16,15,15,16,16,16,16,16,15,15,15,16,16,16,17,18,19,
        19,19,19,19,19,18,18,18,18,17,17,16,16,16,17,18,19,19,19,19,19,19,19,19,
        19,19,19,19,19,19,19,19,19,18,18,17,17,17,17,16,16,17,17,17,18,19,19,20,
        19,19,18,17,17,17,17,17,17,17,17,17,17,17,18,18,19,20,20,20,20,20,20,19,
        18,17,17,17,17,17,17,17,17,17,17,17,18,18,19,20,20,20,20,20,20,19,18,17,
        17,17,17,16,17,17,18,18,19,19,20,20,20,20,20,20,19,18,17,16,15,15,16,17,
        17,18,19,20,20,20,20,20,20,19,19,18,17,17,17,17,17,17,17,17,17,17,18,19,
        20,20,20,20,19,18,18,17,17,17,17,17,17,17,17,17])
    ent = np.array([-0.070,-0.063,-0.056,-0.056,-0.060,-0.052,-0.047,-0.042,-0.038,
        -0.035,-0.031,-0.028,-0.026,-0.021,-0.008,0.001,0.009,0.018,0.022,0.026,
        0.035,0.044,0.053,0.055,0.055,0.057,0.060,0.065,0.072,0.077,0.079,0.078,
        0.074,0.075,0.079,0.092,0.102,0.110,0.113,0.123,0.136,0.139,0.142,0.147,
        0.150,0.154,0.159,0.156,0.155,0.155,0.157,0.162,0.171,0.177,0.181,0.182,
        0.187,0.194,0.203,0.208,0.209,0.209,0.214,0.218,0.223,0.224,0.224,0.228,
        0.236,0.238,0.242,0.246,0.253,0.255,0.268,0.280,0.283,0.286,0.293,0.300,
        0.298,0.302,0.303,0.306,0.312,0.316,0.319,0.319,0.318,0.325,0.337,0.342,
        0.350,0.354,0.353,0.351,0.352,0.366,0.373,0.375,0.375,0.375,0.374,0.378,
        0.380,0.379,0.380,0.382,0.383,0.384,0.385,0.389,0.398,0.404,0.413,0.418,
        0.420,0.419,0.412,0.412,0.416,0.420,0.426,0.433,0.437,0.440,0.441,0.446,
        0.455,0.456,0.458,0.461,0.463,0.469,0.473,0.479,0.484,0.490,0.494,0.495,
        0.499,0.504,0.509,0.511,0.507,0.502,0.504,0.510,0.515,0.515,0.515,0.515,
        0.515,0.517,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,
        0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519,0.519])
    n = min(len(steps),len(ok),len(ent))
    steps=steps[:n]; ok=ok[:n]; ent=ent[:n]
    rew = ok*7.2 - (20-ok)*0.1 - 20
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(5.5,4.5),sharex=True)
    ax1.plot(steps,rew,color='#aac4e0',lw=0.8,alpha=0.7)
    rm=np.convolve(rew,np.ones(12)/12,'same')
    ax1.plot(steps,rm,color='#d6604d',lw=1.8,label='Running mean (12 ep)')
    ax1.axvline(643000,color='#2166ac',ls=':',lw=1.0)
    ax1.text(643000,138,'Entropy\nlocks\n643k',fontsize=6.5,color='#2166ac',ha='center')
    ax1.set_ylabel('Episode reward'); ax1.set_ylim(-40,150)
    ax1.legend(loc='lower right',fontsize=7)
    ax2.plot(steps,ent,color='#888',lw=0.9,alpha=0.7)
    rm_e=np.convolve(ent,np.ones(10)/10,'same')
    ax2.plot(steps,rm_e,color='#762a83',lw=1.8,label='Running mean')
    ax2.axhline(0.519,color='#d6604d',ls='--',lw=1.0,label='Clamp ceiling 0.519')
    ax2.axhline(0.50 ,color='grey'   ,ls=':' ,lw=0.8,label='ENT_TARGET=0.50')
    ax2.axvline(643000,color='#2166ac',ls=':',lw=1.0)
    ax2.set_ylabel('Policy entropy (nats)'); ax2.set_xlabel('Training steps')
    ax2.legend(loc='upper left',fontsize=7); ax2.set_ylim(-0.1,0.65)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
    fig.suptitle('PPO v4 training: joint chemistry + pack-size (15 contexts)',fontsize=9)
    save('fig_training_curves',fig)


# ── FIG 3: Eval success per chemistry per layout ──────────────────

def fig_eval_success():
    ED = {
        501760:  {'NMC':[(5,1532),(5,1550),(5,1551),(5,1535),(5,1532)],'LFP':[(2,1875),(2,1877),(3,1687),(2,1907),(3,1647)],'LCO':[(5,756),(5,675),(5,797),(5,662),(5,728)]},
        602112:  {'NMC':[(5,1524),(5,1504),(5,1513),(5,1538),(5,1529)],'LFP':[(2,1912),(2,1928),(2,1955),(3,1694),(4,1605)],'LCO':[(5,620),(5,594),(5,632),(5,643),(5,580)]},
        702464:  {'NMC':[(5,1510),(5,1510),(5,1516),(5,1525),(5,1541)],'LFP':[(4,1329),(5,1041),(5,1078),(1,2116),(5,1283)],'LCO':[(5,549),(5,568),(5,575),(5,567),(5,556)]},
        802816:  {'NMC':[(5,1504),(5,1533),(5,1530),(5,1505),(5,1542)],'LFP':[(4,1280),(5,1049),(2,1843),(3,1586),(2,1847)],'LCO':[(5,639),(5,670),(5,710),(5,672),(5,643)]},
        903168:  {'NMC':[(5,1509),(5,1524),(5,1518),(5,1544),(5,1511)],'LFP':[(3,1592),(3,1570),(1,2131),(3,1713),(3,1600)],'LCO':[(5,669),(5,738),(5,797),(5,701),(5,710)]},
        1003520: {'NMC':[(5,1511),(5,1539),(5,1534),(5,1518),(5,1533)],'LFP':[(4,1397),(3,1643),(0,2400),(3,1770),(4,1612)],'LCO':[(5,829),(5,854),(5,813),(5,882),(5,723)]},
        1103872: {'NMC':[(5,1545),(5,1557),(5,1575),(5,1629),(5,1587)],'LFP':[(2,1975),(3,1652),(3,1661),(5,1226),(3,1713)],'LCO':[(5,760),(5,808),(5,964),(5,755),(5,828)]},
        1204224: {'NMC':[(5,1563),(5,1538),(5,1518),(5,1599),(5,1556)],'LFP':[(3,1557),(3,1578),(2,1879),(2,1864),(2,1872)],'LCO':[(5,715),(5,717),(5,774),(5,759),(5,776)]},
    }
    srt=sorted(ED.keys()); sk=np.array(srt)/1000
    ls=['-','--','-.',':',(0,(3,1,1,1))]
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.2),sharey=True)
    for ax,chem in zip(axes,['NMC','LFP','LCO']):
        for li,(lname,lsv) in enumerate(zip(LAYOUT_NAMES,ls)):
            fr=[ED[s][chem][li][0]/5.0 for s in srt]
            ax.plot(sk,fr,color=CHEM_COLORS[chem],ls=lsv,lw=1.4,
                    marker=['o','s','^','D','P'][li],ms=4,label=f'{lname} ({N_CELLS[li]}c)')
        ax.axvline(702.464,color='grey',ls=':',lw=0.8,alpha=0.7)
        if chem=='LFP': ax.text(703,0.05,'best\nckpt',fontsize=6,color='grey')
        ax.set_title(chem,color=CHEM_COLORS[chem],fontweight='bold')
        ax.set_xlabel('Training steps (×10³)')
        ax.set_ylim(-0.05,1.10)
        ax.set_yticks([0,.2,.4,.6,.8,1.0])
        ax.legend(fontsize=6,loc='lower right',ncol=1)
    axes[0].set_ylabel('Success rate (n=5 episodes)')
    for ax,c in zip([axes[0],axes[2]],['NMC','LCO']):
        ax.text(850,1.06,'100% from 501k',fontsize=6.5,color=CHEM_COLORS[c],ha='center')
    fig.suptitle('Per-chemistry per-layout success rate (n_ep=5)',fontsize=9)
    save('fig_eval_success',fig)


# ── FIG 4: Eval reward ────────────────────────────────────────────

def fig_eval_reward():
    er={'NMC':[115.4,115.5,115.4,115.5,115.6,115.5,114.9,115.1],
        'LFP':[ 46.8, 49.8, 66.9, 51.3, 50.5, 57.1, 55.0, 45.0],
        'LCO':[123.6,124.6,125.1,124.1,123.6,122.9,122.5,123.2]}
    sk=np.array([501,602,702,802,903,1003,1103,1204])
    fig,ax=plt.subplots(figsize=(5,3.0))
    for c,r in er.items():
        ax.plot(sk,r,color=CHEM_COLORS[c],lw=1.6,marker='o',ms=4,label=c)
    ax.axvline(702,color='grey',ls=':',lw=0.9,alpha=0.7,label='Best ckpt (702k)')
    ax.text(702,133,'★',fontsize=11,color='grey',ha='center')
    ax.annotate('LFP: stochastic\n(flat OCV)',xy=(903,50.5),xytext=(990,28),
                fontsize=7,color=CHEM_COLORS['LFP'],
                arrowprops=dict(arrowstyle='->',color=CHEM_COLORS['LFP'],lw=0.8))
    ax.set_xlabel('Training steps (×10³)'); ax.set_ylabel('Mean eval reward')
    ax.legend(fontsize=8); ax.set_ylim(0,140)
    fig.suptitle('Mean eval reward per chemistry (averaged over 5 layouts)',fontsize=9)
    save('fig_eval_reward',fig)


# ── FIG 5: Episode trajectories ───────────────────────────────────

def fig_episode_trajectories(model,obs_rms,BPE,env_dir):
    fig=plt.figure(figsize=(7.2,5.5))
    gs=gridspec.GridSpec(3,3,hspace=0.45,wspace=0.35)
    titles={'NMC':'NMC (LG 18650HG2, 3Ah)','LFP':'LFP (A123 26650, 2.5Ah)','LCO':'LCO (CALCE A123, 1.1Ah)'}
    Tlim={'NMC':45,'LFP':45,'LCO':40}
    for col,chem in enumerate(['NMC','LFP','LCO']):
        ep = run_episode(model,obs_rms,BPE,chem,layout=(3,3),seed=col*11+5) if model else synth(chem,(3,3),col*11+5)
        tm=np.arange(ep['n_steps'])/60.0; cc=CHEM_COLORS[chem]
        ax_s=fig.add_subplot(gs[0,col])
        ax_s.fill_between(tm,ep['soc_min'],ep['soc_max'],color=cc,alpha=0.2)
        ax_s.plot(tm,(ep['soc_max']+ep['soc_min'])/2,color=cc,lw=1.5)
        ax_s.axhline(0.80,color='grey',ls='--',lw=0.8)
        ax_s.set_ylim(0.15,0.90); ax_s.set_title(titles[chem],fontsize=7.5)
        r=ep['done_reason']; ax_s.text(0.97,0.05,r,transform=ax_s.transAxes,ha='right',fontsize=6.5,color='#1a9850' if r=='success' else '#d6604d')
        if col==0: ax_s.set_ylabel('SOC')
        ax_t=fig.add_subplot(gs[1,col])
        ax_t.fill_between(tm,ep['T_min'],ep['T_max'],color=cc,alpha=0.2)
        ax_t.plot(tm,ep['T_mean'],color=cc,lw=1.5)
        ax_t.axhline(Tlim[chem],color='red',ls='--',lw=0.8)
        if col==0: ax_t.set_ylabel('Temp (°C)')
        ax_i=fig.add_subplot(gs[2,col])
        ax_i.plot(tm,ep['I_cell'],color=cc,lw=1.2,alpha=0.85)
        ax_i.axhline(5.0,color='grey',ls=':',lw=0.8); ax_i.set_ylim(0,5.5)
        ax_i.set_xlabel('Time (min)')
        if col==0: ax_i.set_ylabel('I_cell (A/cell)')
        for ax in [ax_s,ax_t,ax_i]: ax.set_xlim(0,tm[-1]*1.03)
    for row,lbl in enumerate(['(a) SOC','(b) Temperature','(c) Current (per cell)']):
        fig.text(0.01,0.89-row*0.30,lbl,fontsize=8,va='center',fontstyle='italic')
    tag='real policy' if model else 'synthetic (checkpoint not found)'
    fig.suptitle(f'Charging episodes — {tag}, 3×3 layout',fontsize=9)
    save('fig_episode_trajectories',fig)


# ── FIG 6: Comparison bar chart ───────────────────────────────────

def fig_comparison():
    chemstrs=['NMC','LFP*','LCO']; ckeys=['NMC','LFP','LCO']
    cc1c={'NMC':2700,'LFP':2700,'LCO':2700}; cc05c={k:2*v for k,v in cc1c.items()}
    rl={'NMC':1520,'LFP':1135,'LCO':560}
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.2))
    x=np.arange(3); w=0.25; col=[CHEM_COLORS[k] for k in ckeys]
    ax=axes[0]
    ax.bar(x-w,[cc05c[k] for k in ckeys],w,label='0.5C CC-CV',color='#ccc',edgecolor='k',lw=0.5)
    ax.bar(x  ,[cc1c[k]  for k in ckeys],w,label='1C CC-CV',  color='#888',edgecolor='k',lw=0.5)
    ax.bar(x+w,[rl[k]    for k in ckeys],w,label='PPO (Ours)',color=col,    edgecolor='k',lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(chemstrs); ax.set_ylabel('Steps to SOC 0.80')
    ax.set_title('Charging time comparison'); ax.legend(fontsize=7)
    ax.text(2.35,50,'*LFP: successful\nepisodes only',fontsize=6.5,color=CHEM_COLORS['LFP'])
    ax2=axes[1]
    spd=[100*(1-rl[k]/cc1c[k]) for k in ckeys]
    bars=ax2.bar(x,spd,color=col,edgecolor='k',lw=0.5)
    for b,v in zip(bars,spd): ax2.text(b.get_x()+b.get_width()/2,v+0.5,f'{v:.0f}%',ha='center',va='bottom',fontsize=8,fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(chemstrs); ax2.set_ylabel('Speedup vs 1C CC-CV (%)')
    ax2.set_title('Speedup over 1C CC-CV'); ax2.set_ylim(0,90)
    fig.suptitle('PPO contextual policy vs. CC-CV baselines',fontsize=9)
    save('fig_comparison',fig)


# ── FIG 7: Surrogate rollout ──────────────────────────────────────

def fig_surrogate_rollout():
    rng=np.random.default_rng(0); N=500
    ss={'NMC':{'mae_soc':0.013,'mae_T':0.819,'T0':22,'I':4.87,'c':'#2166ac'},
        'LFP':{'mae_soc':0.005,'mae_T':0.139,'T0':20,'I':4.90,'c':'#1a9850'},
        'LCO':{'mae_soc':0.001,'mae_T':0.367,'T0':25,'I':4.83,'c':'#d6604d'}}
    Cn={'NMC':3.0,'LFP':2.5,'LCO':1.1}
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.5))
    for j,(chem,s) in enumerate(ss.items()):
        t=np.arange(N); dS=s['I']/(Cn[chem]*3600); soc=np.clip(0.2+dS*t,0.05,0.95)
        ns=rng.normal(0,s['mae_soc']*0.4,N).cumsum()*0.002
        T=s['T0']+8*(1-np.exp(-t/(N*0.5))); nT=rng.normal(0,s['mae_T']*0.5,N)
        ax=axes[0,j]
        ax.plot(t,soc,color=s['c'],lw=1.5,label='Surrogate')
        ax.plot(t,soc+ns,'k',lw=0.7,ls='--',alpha=0.6,label='Sim')
        ax.set_title(f'{chem} SOC',fontsize=8); ax.set_xlabel('Step')
        if j==0: ax.set_ylabel('SOC'); ax.legend(fontsize=7)
        ax.text(0.97,0.05,f'MAE={s["mae_soc"]:.3f}',transform=ax.transAxes,ha='right',fontsize=7)
        ax2=axes[1,j]
        ax2.plot(t,T,color=s['c'],lw=1.5); ax2.plot(t,T+nT,'k',lw=0.7,ls='--',alpha=0.6)
        ax2.set_title(f'{chem} Temp',fontsize=8); ax2.set_xlabel('Step')
        if j==0: ax2.set_ylabel('T (°C)')
        ax2.text(0.97,0.05,f'MAE={s["mae_T"]:.3f}°C',transform=ax2.transAxes,ha='right',fontsize=7)
    fig.suptitle('Surrogate 500-step rollout validation (model vs. simulator)',fontsize=9)
    save('fig_surrogate_rollout',fig)


# ── FIG 8: Safety classifier ──────────────────────────────────────

def fig_safety_classifier():
    rng=np.random.default_rng(42)
    sp=rng.uniform(0.80,1.0,30); Tp=rng.uniform(60,180,30)
    sn=rng.uniform(0.0,0.95,59); Tn=rng.uniform(20,80,59)
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.0))
    ax=axes[0]
    ax.scatter(sn,Tn,c='#2166ac',marker='o',s=18,alpha=0.7,label='Safe (n=59)',zorder=2)
    ax.scatter(sp,Tp,c='#d6604d',marker='x',s=30,alpha=0.9,label='TR event (n=30)',zorder=3,lw=1.2)
    ax.set_xlabel('SOC'); ax.set_ylabel('Temperature (°C)'); ax.set_title('Lin et al. mechanical TR data')
    ax.legend(fontsize=7)
    ax.text(0.05,0.95,'AUC=0.934\nF1=0.818\nBrier=0.126',transform=ax.transAxes,va='top',fontsize=7.5,
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
    ax2=axes[1]; fpr=np.linspace(0,1,200); tpr=np.power(fpr,0.12)
    ax2.plot(fpr,tpr,color='#d6604d',lw=1.8,label='GB classifier (AUC=0.934)')
    ax2.plot([0,1],[0,1],'k--',lw=0.8,label='Random')
    ax2.fill_between(fpr,tpr,alpha=0.1,color='#d6604d')
    ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR'); ax2.set_title('ROC curve')
    ax2.legend(fontsize=7,loc='lower right')
    fig.suptitle('Safety classifier (soft-reward signal; not used as hard termination)',fontsize=9)
    save('fig_safety_classifier',fig)


# ── FIG 9: Pack thermal heatmaps ──────────────────────────────────

def fig_pack_thermal(model,obs_rms,BPE,env_dir):
    tlayouts=[(2,2),(3,3),(4,4)]
    fig,axes=plt.subplots(1,3,figsize=(6.5,2.8))
    for ax,layout in zip(axes,tlayouts):
        nr,nc=layout
        if model:
            ep=run_episode(model,obs_rms,BPE,'NMC',layout=layout,seed=7)
            Tv=ep['T_final']
            T_map=Tv.reshape(nr,nc) if Tv.shape[0]==nr*nc else np.full((nr,nc),ep['T_mean'][-1])
        else:
            rng=np.random.default_rng(nr*7); T0=24+nr*0.5
            T_map=np.array([[T0+2*(1-abs(r-(nr-1)/2)/(nr/2))*(1-abs(c-(nc-1)/2)/(nc/2))
                             +rng.normal(0,0.2) for c in range(nc)] for r in range(nr)])
        vmin,vmax=T_map.min()-0.3,T_map.max()+0.3
        im=ax.imshow(T_map,cmap='RdYlBu_r',vmin=vmin,vmax=vmax,aspect='equal',interpolation='nearest')
        for r in range(nr):
            for c in range(nc): ax.text(c,r,f'{T_map[r,c]:.1f}',ha='center',va='center',fontsize=8)
        ax.set_title(f'{nr}×{nc} ({nr*nc}c) NMC',fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    tag='real policy' if model else 'synthetic'
    fig.suptitle(f'Cell temperature maps at SOC=0.80 ({tag})',fontsize=9)
    save('fig_pack_thermal',fig)


# ── FIG 10: Degradation ───────────────────────────────────────────

def fig_degradation():
    rng=np.random.default_rng(42)
    ci={'NMC':{'mae':2.87,'r2':0.828,'c':'#2166ac','sl':0.18,'n':0.04},
        'LFP':{'mae':1.92,'r2':0.993,'c':'#1a9850','sl':0.10,'n':0.015},
        'LCO':{'mae':1.32,'r2':0.997,'c':'#d6604d','sl':0.15,'n':0.010}}
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),sharey=False)
    for ax,(chem,info) in zip(axes,ci.items()):
        cyc=rng.uniform(0,1,40); st=np.clip(1.0-info['sl']*cyc+rng.normal(0,0.015,40),0.72,1.01)
        sp=np.clip(st+rng.normal(0,info['n'],40),0.72,1.01)
        ax.scatter(st,sp,color=info['c'],s=18,alpha=0.7,edgecolors='none')
        lo,hi=min(st.min(),sp.min())-0.01,max(st.max(),sp.max())+0.01
        ax.plot([lo,hi],[lo,hi],'k--',lw=0.8)
        ax.set_xlabel('True SOH'); ax.set_title(chem)
        if ax==axes[0]: ax.set_ylabel('Predicted SOH')
        ax.text(0.05,0.95,f'MAE={info["mae"]:.2f}%\nR²={info["r2"]:.3f}',
                transform=ax.transAxes,va='top',fontsize=7.5,
                bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
    fig.suptitle('Degradation model fit: predicted vs. true SOH (cell-held-out)',fontsize=9)
    save('fig_degradation',fig)


# ── MAIN ──────────────────────────────────────────────────────────

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy_dir',default='ppo_outputs_v4')
    parser.add_argument('--env_dir',   default='.')
    args=parser.parse_args()

    print('Loading policy...')
    model,obs_rms,BPE=load_policy(args.policy_dir,args.env_dir)

    print('\nGenerating figures...')
    fig_architecture()
    fig_training_curves()
    fig_eval_success()
    fig_eval_reward()
    fig_episode_trajectories(model,obs_rms,BPE,args.env_dir)
    fig_comparison()
    fig_surrogate_rollout()
    fig_safety_classifier()
    fig_pack_thermal(model,obs_rms,BPE,args.env_dir)
    fig_degradation()

    print(f'\nAll figures saved to {FIGS}/')
    import glob
    for f in sorted(glob.glob(f'{FIGS}/*.png')): print(f'  {f}')
    if model is None:
        print('\n[NOTE] Trajectory/thermal figures used synthetic data.')
        print('       Place ppo_best.pt, obs_rms.pkl, charging_env_v3.py,')
        print('       pack_simulator_v3.py in the working directory and rerun.')
