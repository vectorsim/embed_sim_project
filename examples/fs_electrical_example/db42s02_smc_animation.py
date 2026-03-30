#!/usr/bin/env python3
"""
db42s02_smc_tune_and_animate.py
================================
Full pipeline — single entry point, reusable:
  1. BASELINE simulation   (original _DB42S02 gains)
  2. Bayesian GP tuner     (20 iterations, 5-D)
  3. TUNED simulation      (best gains)
  4. Animate TUNED result  on (id, iq, RPM) manifold

Figure layout  (24 × 11, fixed camera)
---------------------------------------
  LEFT  65%  : 3-D manifold  id × iq × RPM  (LARGE)
                 Grey static  = baseline path
                 Coloured     = tuned path (SMC phase)
                   RED    |s|>80   REACHING
                   YELLOW |s|>15   SLIDING
                   GREEN  |s|≤15   LOCKED
                   BLUE           TRANSIENT (load step)

  RIGHT 35%, 3 rows (small):
    [0] Speed RPM  : ref (white dashed) + baseline (grey) + tuned (cyan)
    [1] id / iq    : baseline grey, tuned id=blue iq=orange  (live)
    [2] Tuner cost : all 20 iters static, animated white cursor

Outputs
-------
  db42s02_smc_tune_and_animate.gif
  db42s02_smc_tune_and_animate.png
  smc_tuner_best.py
"""

from __future__ import annotations

import sys, os, math, time, contextlib, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["EMBEDSIM_SILENT"] = "1"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm as sp_norm

from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE    = get_current_parent()
_ROOT    = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.simulation_engine import VectorDelay

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block          import SVPWMBlock
from PMSM_Plant_FMUBlock  import PMSM_Plant_FMUBlock
from smc_controller_block import SMCControllerBlock, _DB42S02

_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Plant_FMU.fmu")

# =============================================================================
# Constants
# =============================================================================
V_DC        = _DB42S02.SMC_V_DC
TARGET_RPM  = 2000.0
TARGET_RADS = TARGET_RPM * 2.0 * math.pi / 60.0
T_SIM       = 2.0
DT          = 50e-6
T_LOAD_T1   = 0.5
T_LOAD_T2   = 1.2
T_LOAD_ZERO  = 0.000
T_LOAD_LIGHT = 0.005
T_LOAD_HEAVY = 0.020
_MOTOR_OUT_SIZE = 8
_IDX_RPM=0; _IDX_IA=1; _IDX_IB=2; _IDX_IC=3; _IDX_THETA_M=4

KT        = _DB42S02.SMC_KT
B_FRIC    = _DB42S02.SMC_B_FRICTION
I_MAX     = _DB42S02.SMC_I_MAX
SVPWM_GAIN= _DB42S02.SMC_SVPWM_GAIN

# Tuner search space
DEFAULTS = np.array([
    _DB42S02.SMC_KS_W,
    _DB42S02.SMC_PHI_W,
    _DB42S02.SMC_KS_I * SVPWM_GAIN,
    _DB42S02.SMC_PHI_I,
    _DB42S02.SMC_LAMBDA_W,
], dtype=np.float64)

PARAM_NAMES = ["KS_W","PHI_W","KS_I_raw","PHI_I","LAMBDA_W"]
PARAM_UNITS = ["A","rad/s","V","A","rad/s"]

BOUNDS = np.column_stack([
    np.maximum(DEFAULTS * 0.50, np.array([0.5,   80.0, 0.10, 0.05, 15.0])),
    np.minimum(DEFAULTS * 1.80, np.array([8.0, 1200.0, 3.00, 2.00, 250.0])),
])

W_BUMP=1.0; W_SS=4.0; W_ID=25.0; W_CHAT=6.0
N_ITER=20;  PENALTY=5000.0

# Phase colours
S_REACH=80.0;  C_REACH="#ff2222"
S_SLIDE=15.0;  C_SLIDE="#ffdd00"
C_LOCK="#22dd22"; C_TRANS="#2288ff"; ID_TRANS=0.4


# =============================================================================
# Silence helper
# =============================================================================
@contextlib.contextmanager
def _silent():
    with open(os.devnull,"w") as dn:
        o,e = sys.stdout, sys.stderr
        sys.stdout=dn; sys.stderr=dn
        try: yield
        finally: sys.stdout=o; sys.stderr=e


# =============================================================================
# Plant + CtrlPacker
# =============================================================================
class DB42S02PlantBlock(PMSM_Plant_FMUBlock):
    TOPO_CATEGORY="plant"; C_CODEGEN_EXCLUDE=True
    def __init__(self,name,fmu_path):
        super().__init__(name=name,fmu_path=fmu_path)
    def compute_py(self,t,dt,input_values=None):
        ta=tb=tc=0.5
        if input_values and input_values[0] is not None:
            v=input_values[0].value
            if len(v)>=3: ta,tb,tc=float(v[0]),float(v[1]),float(v[2])
        # Constant full load — no steps, clean manifold convergence
        tl=T_LOAD_HEAVY
        ta=max(0.05,min(0.95,ta)); tb=max(0.05,min(0.95,tb))
        tc=max(0.05,min(0.95,tc))
        if abs(ta-tb)<1e-6 and abs(tb-tc)<1e-6: ta+=1e-4
        return super().compute_py(t,dt,[VectorSignal(
            np.array([ta,tb,tc,V_DC,tl],dtype=DEFAULT_DTYPE))])
    def compute(self,t,dt,input_values=None):
        return self.compute_py(t,dt,input_values)

class CtrlPacker(VectorBlock):
    C_CODEGEN_EXCLUDE=True
    def __init__(self,name="ctrl_packer",**kw): super().__init__(name,**kw)
    def compute_py(self,t,dt,input_values=None):
        m=(input_values[0].value if input_values and len(input_values)>0
           else np.zeros(_MOTOR_OUT_SIZE,dtype=DEFAULT_DTYPE))
        r=(input_values[1].value if input_values and len(input_values)>1
           else np.zeros(1,dtype=DEFAULT_DTYPE))
        self.output=VectorSignal(np.array([
            float(r[0])             if len(r)>0             else 0.0,
            float(m[_IDX_THETA_M]) if len(m)>_IDX_THETA_M else 0.0,
            float(m[_IDX_IA])      if len(m)>_IDX_IA      else 0.0,
            float(m[_IDX_IB])      if len(m)>_IDX_IB      else 0.0,
            float(m[_IDX_IC])      if len(m)>_IDX_IC      else 0.0,
        ],dtype=DEFAULT_DTYPE),self.name)
        return self.output
    def compute(self,t,dt,input_values=None):
        return self.compute_py(t,dt,input_values)


# =============================================================================
# Simulation helper
# =============================================================================
def _build_gains(params):
    KS_W,PHI_W,KS_I_raw,PHI_I,LAMBDA_W = params
    return dict(SMC_KS_W=KS_W, SMC_PHI_W=PHI_W,
                SMC_KS_I=KS_I_raw/SVPWM_GAIN,
                SMC_PHI_I=PHI_I, SMC_LAMBDA_W=LAMBDA_W)

def run_sim(gains: dict) -> dict | None:
    try:
        with _silent():
            sr  = VectorStep("speed_ref",step_time=0.0,
                             before_value=TARGET_RADS,after_value=TARGET_RADS)
            mot = DB42S02PlantBlock("motor",fmu_path=_FMU_PATH)
            md  = VectorDelay("motor_delay",initial=[0.0]*_MOTOR_OUT_SIZE)
            cp  = CtrlPacker("ctrl_packer")
            smc = SMCControllerBlock("smc",dt_s=DT,**gains)
            sp  = SVPWMPackBlock("svpwm_pack",v_dc=V_DC)
            sp.NUM_INPUTS=1
            sv  = SVPWMBlock("svpwm")
            sk  = VectorEnd("sink")
            mot>>md; md>>cp; sr>>cp; cp>>smc; smc>>sp; sp>>sv; sv>>mot; mot>>sk
            sim=EmbedSim(sinks=[sk],T=T_SIM,dt=DT,solver=ODESolver.EULER)
            sim.run()
    except Exception:
        return None

    ld=smc.log_data
    t  =np.asarray(ld["t"],        dtype=np.float64)
    rpm=np.asarray(ld["speed"],    dtype=np.float64)
    ref=np.asarray(ld["speed_ref"],dtype=np.float64)
    idd=np.asarray(ld["id"],       dtype=np.float64)
    iqq=np.asarray(ld["iq"],       dtype=np.float64)
    iqr=np.asarray(ld["iq_ref"],   dtype=np.float64)

    lam=gains["SMC_LAMBDA_W"]
    err=(ref-rpm)*2.0*math.pi/60.0
    ie =np.zeros_like(err)
    for k in range(1,len(err)):
        ie[k]=ie[k-1]+0.5*(t[k]-t[k-1])*(err[k]+err[k-1])
    s=err+lam*ie
    return {"t":t,"rpm":rpm,"ref":ref,"id":idd,"iq":iqq,"iq_ref":iqr,"s":s}


# =============================================================================
# Cost from sim dict
# =============================================================================
def cost_metrics(d):
    if d is None: return None
    t=d["t"]; rpm=d["rpm"]; idd=d["id"]; iqr=d["iq_ref"]
    if len(t)<50 or np.max(np.abs(rpm))>TARGET_RPM*2.5: return None
    def bump(ts,win=0.25):
        pre=rpm[t<ts]; post=rpm[(t>=ts)&(t<ts+win)]
        if len(pre)<5 or len(post)<5: return 0.0
        return max(0.0,float(np.mean(pre[-20:]))-float(np.min(post)))
    ss=t>0.80*T_SIM
    if not np.any(ss): return None
    br=(bump(T_LOAD_T1)+bump(T_LOAD_T2))/2.0
    sse=float(np.mean(np.abs(rpm[ss]-TARGET_RPM)))
    idr=float(np.sqrt(np.mean(idd[ss]**2)))
    iqc=float(np.std(iqr[ss]))
    cost=W_BUMP*br+W_SS*sse+W_ID*idr+W_CHAT*iqc
    return {"cost":cost,"bump":br,"ss_error":sse,"id_rms":idr,"iq_chat":iqc}


# =============================================================================
# Latin Hypercube
# =============================================================================
def latin_hypercube(n,bounds,rng,centre=None):
    ndim=len(bounds); lo,hi=bounds[:,0],bounds[:,1]
    S=np.zeros((n,ndim))
    start=0
    if centre is not None:
        S[0]=np.clip(centre,lo,hi); start=1
    n_lhs=n-start
    if n_lhs>0:
        cuts=np.linspace(0,1,n_lhs+1)
        for d in range(ndim):
            u=rng.uniform(cuts[:-1],cuts[1:]); rng.shuffle(u)
            S[start:,d]=lo[d]+u*(hi[d]-lo[d])
    return np.clip(S,lo,hi)


# =============================================================================
# GP Expected Improvement
# =============================================================================
def next_candidate(X_obs,y_obs,bounds,rng,n_cand=300):
    lo,hi=bounds[:,0],bounds[:,1]
    Xn=(X_obs-lo)/(hi-lo+1e-12)
    kernel=ConstantKernel(1.0,(0.1,10.0))*Matern(
        length_scale=1.0,length_scale_bounds=(0.1,10.0),nu=2.5)
    gp=GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5,
                                normalize_y=True,alpha=1e-4)
    gp.fit(Xn,y_obs)
    Xc=rng.uniform(0,1,size=(n_cand,bounds.shape[0]))
    mu,sig=gp.predict(Xc,return_std=True)
    sig=np.maximum(sig,1e-9)
    yb=np.min(y_obs); Z=(yb-mu-0.01)/sig
    ei=(yb-mu-0.01)*sp_norm.cdf(Z)+sig*sp_norm.pdf(Z)
    ei[sig<1e-8]=0.0
    return lo+Xc[np.argmax(ei)]*(hi-lo)


# =============================================================================
# TUNER
# =============================================================================
def run_tuner():
    rng=np.random.default_rng(seed=42)
    print("="*65)
    print("  SMC Gain Tuner — Bayesian GP  |  20 iterations")
    print(f"  Baseline: "+
          "  ".join(f"{n}={v:.4f}{u}"
                    for n,u,v in zip(PARAM_NAMES,PARAM_UNITS,DEFAULTS)))
    print("="*65)

    history=[]; X_obs=[]; y_obs=[]
    lhs_pts=latin_hypercube(6,BOUNDS,rng,centre=DEFAULTS)

    for i,params in enumerate(lhs_pts):
        phase="baseline" if i==0 else f"LHS-{i}"
        gains=_build_gains(params)
        print(f"\n[{i:2d}/{N_ITER-1}] {phase}")
        print("         "+"  ".join(f"{n}={v:.4f}{u}"
              for n,u,v in zip(PARAM_NAMES,PARAM_UNITS,params)))
        t0=time.perf_counter()
        d=run_sim(gains); m=cost_metrics(d)
        elapsed=time.perf_counter()-t0
        if m is None:
            cost=PENALTY
            m={"cost":cost,"bump":999,"ss_error":999,"id_rms":99,"iq_chat":99}
            print(f"         UNSTABLE  ({elapsed:.1f}s)")
        else:
            cost=m["cost"]
            print(f"         cost={cost:.2f}  bump={m['bump']:.1f}RPM"
                  f"  ss={m['ss_error']:.2f}RPM  id={m['id_rms']:.3f}A"
                  f"  ({elapsed:.1f}s)")
        m["params"]=params.copy(); history.append(m)
        X_obs.append(params); y_obs.append(cost)

    for i in range(6,N_ITER):
        params=next_candidate(np.array(X_obs),np.array(y_obs),BOUNDS,rng)
        gains=_build_gains(params)
        print(f"\n[{i:2d}/{N_ITER-1}] GP-EI")
        print("         "+"  ".join(f"{n}={v:.4f}{u}"
              for n,u,v in zip(PARAM_NAMES,PARAM_UNITS,params)))
        t0=time.perf_counter()
        d=run_sim(gains); m=cost_metrics(d)
        elapsed=time.perf_counter()-t0
        if m is None:
            cost=PENALTY
            m={"cost":cost,"bump":999,"ss_error":999,"id_rms":99,"iq_chat":99}
            print(f"         UNSTABLE  ({elapsed:.1f}s)")
        else:
            cost=m["cost"]
            print(f"         cost={cost:.2f}  bump={m['bump']:.1f}RPM"
                  f"  ss={m['ss_error']:.2f}RPM  id={m['id_rms']:.3f}A"
                  f"  ({elapsed:.1f}s)")
        m["params"]=params.copy(); history.append(m)
        X_obs.append(params); y_obs.append(cost)

    costs=[h["cost"] for h in history]
    best_idx=int(np.argmin(costs))
    best=history[best_idx]
    best_params=best["params"]
    best_gains=_build_gains(best_params)

    print("\n"+"="*65)
    print("  TUNING COMPLETE")
    print("="*65)
    print(f"  Baseline (iter  0): cost={history[0]['cost']:.2f}"
          f"  bump={history[0]['bump']:.1f}RPM")
    print(f"  Best     (iter {best_idx:2d}): cost={best['cost']:.2f}"
          f"  bump={best['bump']:.1f}RPM")
    print()
    for name,unit,d,b in zip(PARAM_NAMES,PARAM_UNITS,DEFAULTS,best_params):
        delta=(b-d)/d*100.0
        print(f"    {name:12s}: {d:.4f} -> {b:.4f} {unit:6s}"
              f"  ({'UP' if b>d else 'DN'} {abs(delta):.1f}%)")
    base_bump=history[0]["bump"]; best_bump=best["bump"]
    pct=(base_bump-best_bump)/base_bump*100.0 if base_bump>0.1 else 0
    print(f"\n  Bump: {base_bump:.1f} -> {best_bump:.1f} RPM  ({pct:.1f}% reduction)")
    print("="*65)

    # Write best gains file
    KS_W,PHI_W,KS_I_raw,PHI_I,LAMBDA_W=best_params
    KS_I=KS_I_raw/SVPWM_GAIN
    Path("smc_tuner_best.py").write_text(f"""\
# smc_tuner_best.py — auto-generated
# Best iter {best_idx}  cost={best['cost']:.3f}  bump={best['bump']:.1f}RPM
TUNED_KS_W     = {KS_W:.6f}    # A       (was {_DB42S02.SMC_KS_W:.6f})
TUNED_PHI_W    = {PHI_W:.6f}   # rad/s   (was {_DB42S02.SMC_PHI_W:.6f})
TUNED_KS_I     = {KS_I:.6f}    # V/SVPWM (was {_DB42S02.SMC_KS_I:.6f})
TUNED_KS_I_RAW = {KS_I_raw:.6f}   # V physical
TUNED_PHI_I    = {PHI_I:.6f}   # A       (was {_DB42S02.SMC_PHI_I:.6f})
TUNED_LAMBDA_W = {LAMBDA_W:.6f}  # rad/s (was {_DB42S02.SMC_LAMBDA_W:.6f})
""")
    print("[Gains] smc_tuner_best.py")
    return history, best_idx, best_gains


# =============================================================================
# Phase colour per sample
# =============================================================================
def phase_colours(s_arr, id_arr):
    out=[]
    for s,id_v in zip(s_arr,id_arr):
        a=abs(s)
        if   a>S_REACH:          out.append(C_REACH)
        elif a>S_SLIDE:          out.append(C_SLIDE)
        elif abs(id_v)>ID_TRANS: out.append(C_TRANS)
        else:                    out.append(C_LOCK)
    return out


# =============================================================================
# RPM manifold surface
# =============================================================================
def build_surface(id_range,iq_range):
    ID,IQ=np.meshgrid(id_range,iq_range)
    RPM=np.clip((KT*IQ-T_LOAD_HEAVY)/(B_FRIC+1e-12)*60/(2*math.pi),
                0.0,TARGET_RPM*1.15)
    return ID,IQ,RPM


# =============================================================================
# ANIMATION
# =============================================================================
def make_animation(base, tuned, history, best_idx,
                   gif_path="db42s02_smc_tune_and_animate.gif",
                   png_path="db42s02_smc_tune_and_animate.png"):

    # Phase colours on tuned path
    cols_all=phase_colours(tuned["s"],tuned["id"])

    # Downsample to ~200 frames
    step=max(1,len(tuned["t"])//200)
    fx=np.arange(0,len(tuned["t"]),step)
    t_a  =tuned["t"][fx];   id_a =tuned["id"][fx]
    iq_a =tuned["iq"][fx];  rpm_a=tuned["rpm"][fx]
    ref_a=tuned["ref"][fx]; s_a  =tuned["s"][fx]
    cols_a=[cols_all[i] for i in fx]
    # Baseline interpolated
    rpm_b=np.interp(t_a,base["t"],base["rpm"])
    id_b =np.interp(t_a,base["t"],base["id"])
    iq_b =np.interp(t_a,base["t"],base["iq"])

    # Manifold surface
    id_pad=max(0.9,max(np.max(np.abs(tuned["id"])),
                       np.max(np.abs(base["id"])))*1.7)
    iq_pad=max(0.5,max(np.max(tuned["iq"]),np.max(base["iq"]))*1.25)
    id_rng=np.linspace(-id_pad,id_pad,55)
    iq_rng=np.linspace(-0.2,iq_pad,55)
    ID,IQ,RPM_SURF=build_surface(id_rng,iq_rng)
    rpm_max_s=TARGET_RPM*1.15
    surf_norm=np.clip(RPM_SURF/rpm_max_s,0,1)
    mtpa_iq=np.linspace(0,I_MAX,80)
    mtpa_rpm=np.clip((KT*mtpa_iq-T_LOAD_HEAVY)/(B_FRIC+1e-12)*60/(2*math.pi),
                     0,rpm_max_s)

    # Tuner data
    tc_iters=np.arange(N_ITER)
    tc_costs=np.array([h["cost"] for h in history])
    tc_bumps=np.array([h["bump"] for h in history])
    tc_best =np.minimum.accumulate(tc_costs)

    base_bump=history[0]["bump"]
    best_bump=history[best_idx]["bump"]
    pct=(base_bump-best_bump)/base_bump*100.0 if base_bump>0.1 else 0

    # ── Figure ──────────────────────────────────────────────────────────────
    fig=plt.figure(figsize=(24,11),facecolor="#080808")
    fig.suptitle(
        f"SMC FOC  |  Bayesian GP Tuner  |  NANOTEC DB42S02  |  20 kHz"
        f"        Bump: {base_bump:.0f} → {best_bump:.0f} RPM  (−{pct:.0f}%)",
        color="white",fontsize=15,fontweight="bold",y=0.976)

    gs=GridSpec(3,2,
                left=0.01,right=0.985,bottom=0.06,top=0.93,
                wspace=0.22,hspace=0.52,
                width_ratios=[1.85,1.0],
                height_ratios=[1.0,1.0,1.0])

    ax3d =fig.add_subplot(gs[:,0],projection="3d",facecolor="#0c0c0c")
    ax_sp=fig.add_subplot(gs[0,1],facecolor="#0f0f0f")
    ax_iq=fig.add_subplot(gs[1,1],facecolor="#0f0f0f")
    ax_tu=fig.add_subplot(gs[2,1],facecolor="#0f0f0f")

    # 3D style
    ax3d.set_facecolor("#0c0c0c")
    for pane in (ax3d.xaxis.pane,ax3d.yaxis.pane,ax3d.zaxis.pane):
        pane.fill=False; pane.set_edgecolor("#181818")
    ax3d.tick_params(colors="#cccccc",labelsize=11)
    for lb in (ax3d.xaxis.label,ax3d.yaxis.label,ax3d.zaxis.label):
        lb.set_color("#ffffff")
    ax3d.grid(alpha=0.08)

    def style(ax,xl,yl,title):
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#cccccc",labelsize=11)
        ax.xaxis.label.set_color("#ffffff"); ax.yaxis.label.set_color("#ffffff")
        for sp in ax.spines.values(): sp.set_edgecolor("#333333")
        ax.set_xlabel(xl,fontsize=12); ax.set_ylabel(yl,fontsize=12)
        ax.set_title(title,color="white",fontsize=12,fontweight="bold",pad=5)
        ax.grid(alpha=0.18,color="#2a2a2a")

    style(ax_sp,"Time [s]","RPM","Speed  [white=ref  grey=baseline  cyan=tuned]")
    style(ax_iq,"Time [s]","Current [A]","Currents  [blue=id  orange=iq  grey=baseline]")
    style(ax_tu,"Iteration","Cost / Bump [RPM]",
          f"GP Tuner — 20 iters  (best = iter {best_idx})")

    # ── Static 3D ───────────────────────────────────────────────────────────
    ax3d.plot_surface(ID,IQ,RPM_SURF,
                      facecolors=cm.Blues(surf_norm*0.65+0.15),
                      alpha=0.20,linewidth=0,antialiased=True,zorder=1)
    ax3d.contour(ID,IQ,RPM_SURF,
                 levels=np.arange(250,int(rpm_max_s),250),zdir="z",
                 colors=["#1a3a55"],linewidths=0.5,alpha=0.55)
    ax3d.plot(np.zeros_like(mtpa_iq),mtpa_iq,mtpa_rpm,
              color="lime",lw=2.2,ls="--",alpha=0.85,zorder=5,
              label="MTPA  id=0")
    iq_w=np.array([-0.2,iq_pad]); rpm_w=np.array([0.0,rpm_max_s])
    IQW,RPMW=np.meshgrid(iq_w,rpm_w)
    ax3d.plot_surface(np.zeros_like(IQW),IQW,RPMW,
                      color="#223344",alpha=0.07,
                      linewidth=0,antialiased=False,zorder=0)
    ax3d.contour(ID,IQ,RPM_SURF,levels=[TARGET_RPM],zdir="z",
                 colors=["#ff3333"],linewidths=2.0,alpha=0.80)
    ax3d.plot_wireframe(ID,IQ,np.zeros_like(RPM_SURF),
                        color="#111820",linewidth=0.3,alpha=0.4,zorder=0)
    # Full baseline path (static grey, thicker for contrast)
    ax3d.plot(base["id"],base["iq"],base["rpm"],
              color="#888888",lw=2.2,alpha=0.60,zorder=3,label="Baseline")

    ax3d.set_xlabel("id  [A]",fontsize=12,labelpad=6)
    ax3d.set_ylabel("iq  [A]",fontsize=12,labelpad=6)
    ax3d.set_zlabel("RPM",    fontsize=12,labelpad=6)
    ax3d.set_zlim(0,rpm_max_s)
    ax3d.set_title("(id, iq, RPM)  Operating Manifold",
                   color="white",fontsize=13,fontweight="bold",pad=10)
    ax3d.view_init(elev=28,azim=-55)

    leg_elems=[
        Line2D([0],[0],color="#888888",lw=3,                label="Baseline"),
        Line2D([0],[0],color=C_REACH,  lw=4, ls="dotted",  label=f"REACHING  |s|>{S_REACH:.0f}  (not converged)"),
        Line2D([0],[0],color=C_SLIDE,  lw=4,               label=f"SLIDING   |s|>{S_SLIDE:.0f}  (converging)"),
        Line2D([0],[0],color=C_LOCK,   lw=4,               label="LOCKED    |s|≤15  (on manifold)"),
        Line2D([0],[0],color=C_TRANS,  lw=4,               label="TRANSIENT  load step"),
        Line2D([0],[0],color="lime",   lw=2.5,ls="--",     label="MTPA  id=0"),
        Line2D([0],[0],color="#ff3333",lw=2,               label=f"{TARGET_RPM:.0f} RPM target"),
    ]
    ax3d.legend(handles=leg_elems,loc="lower left",fontsize=10,
                labelcolor="white",facecolor="#141414",
                edgecolor="#444444",framealpha=0.90)

    # ── Static small panels ─────────────────────────────────────────────────
    # Speed
    ax_sp.plot(base["t"], base["rpm"], color="#888888",lw=1.5,alpha=0.55,
               label="baseline")
    ax_sp.plot(tuned["t"],tuned["ref"],color="white",  lw=1.2,ls="--",
               alpha=0.65,label="ref")
    ax_sp.plot(tuned["t"],tuned["rpm"],color="#224488",lw=0.8,alpha=0.20)
    ax_sp.axhline(TARGET_RPM,color="#ff4444",ls="--",lw=1.0,alpha=0.50)
    ax_sp.axvline(T_LOAD_T1, color="orange", ls=":",lw=1.0,alpha=0.45)
    ax_sp.axvline(T_LOAD_T2, color="red",    ls=":",lw=1.0,alpha=0.45)
    ax_sp.set_xlim(tuned["t"][0],tuned["t"][-1])
    ax_sp.set_ylim(-50,TARGET_RPM*1.10)
    ax_sp.legend(loc="lower right",fontsize=10,labelcolor="white",
                 facecolor="#141414",edgecolor="#2e2e2e",framealpha=0.85)

    # Currents
    ax_iq.plot(base["t"], base["id"], color="#5555aa",lw=1.2,alpha=0.45,
               label="id  baseline")
    ax_iq.plot(base["t"], base["iq"], color="#aa5555",lw=1.2,alpha=0.45,
               label="iq  baseline")
    ax_iq.plot(tuned["t"],tuned["id"],color="#2244aa",lw=0.8,alpha=0.18)
    ax_iq.plot(tuned["t"],tuned["iq"],color="#aa6622",lw=0.8,alpha=0.18)
    ax_iq.axhline(0,     color="#444444",ls="--",lw=0.8)
    ax_iq.axhline( I_MAX,color="#334433",ls=":", lw=0.7,alpha=0.6)
    ax_iq.axhline(-I_MAX,color="#334433",ls=":", lw=0.7,alpha=0.6)
    ax_iq.axvline(T_LOAD_T1,color="orange",ls=":",lw=1.0,alpha=0.45)
    ax_iq.axvline(T_LOAD_T2,color="red",   ls=":",lw=1.0,alpha=0.45)
    ax_iq.set_xlim(tuned["t"][0],tuned["t"][-1])
    ax_iq.set_ylim(-I_MAX*1.2,I_MAX*1.2)

    # Tuner (fully static)
    ax_tu.plot(tc_iters,tc_costs,"o-",color="C0",lw=1.5,ms=5,
               label="cost J",alpha=0.85)
    ax_tu.plot(tc_iters,tc_best, "r--",lw=2.0,label="running best")
    ax_tu2=ax_tu.twinx()
    ax_tu2.plot(tc_iters,tc_bumps,"s-",color="C1",lw=1.2,ms=4,
                alpha=0.80,label="bump RPM")
    ax_tu2.set_ylabel("Bump [RPM]",color="C1",fontsize=11)
    ax_tu2.tick_params(colors="C1",labelsize=10)
    ax_tu.axvline(5,          color="#445566",ls=":", lw=1.2,alpha=0.7)
    ax_tu.axvline(best_idx,   color="gold",   ls="--",lw=2.0,
                  label=f"best = iter {best_idx}")
    ax_tu.set_xlim(-0.5,N_ITER-0.5)
    ax_tu.set_ylabel("Cost J",fontsize=11,color="C0")
    ax_tu.tick_params(colors="C0",labelsize=10)
    ax_tu.text(2.5, max(tc_costs)*0.92,"LHS",  color="#88bbcc",fontsize=11,ha="center",fontweight="bold")
    ax_tu.text(12.5,max(tc_costs)*0.92,"GP-EI",color="#aaccdd",fontsize=11,ha="center",fontweight="bold")
    h1,l1=ax_tu.get_legend_handles_labels()
    h2,l2=ax_tu2.get_legend_handles_labels()
    ax_tu.legend(h1+h2,l1+l2,fontsize=10,labelcolor="white",
                 facecolor="#141414",edgecolor="#2e2e2e",
                 framealpha=0.88,loc="upper right")

    # ── Animated artists ────────────────────────────────────────────────────
    # 3D
    live_dot,   =ax3d.plot([],[],[],"o",ms=10,zorder=9,
                           markeredgecolor="white",markeredgewidth=0.8)
    drop_line,  =ax3d.plot([],[],[],lw=0.9,ls=":",alpha=0.5,
                           color="#aaaaaa",zorder=3)
    shadow_line,=ax3d.plot([],[],[],lw=0.8,alpha=0.28,
                           color="#556677",zorder=2)
    # Speed panel
    sp_live,=ax_sp.plot([],[],lw=2.2,color="cyan",zorder=4,label="tuned")
    sp_dot, =ax_sp.plot([],[],  "o",ms=7,color="cyan",zorder=5)
    # Current panel
    id_live,=ax_iq.plot([],[],lw=2.2,color="#4488ff",zorder=4,label="id  tuned")
    iq_live,=ax_iq.plot([],[],lw=2.2,color="#ff9944",zorder=4,label="iq  tuned")
    id_dot, =ax_iq.plot([],[],  "o",ms=7,color="#4488ff",zorder=5)
    iq_dot, =ax_iq.plot([],[],  "o",ms=7,color="#ff9944",zorder=5)
    # Tuner cursor
    tu_cur, =ax_tu.plot([],[],lw=2.5,color="white",alpha=0.85,zorder=5)

    ax_sp.legend(loc="lower right",fontsize=10,labelcolor="white",
                 facecolor="#141414",edgecolor="#2e2e2e",framealpha=0.88)
    ax_iq.legend(loc="upper right",fontsize=10,labelcolor="white",
                 facecolor="#141414",edgecolor="#2e2e2e",framealpha=0.88)

    # Phase text + time stamp
    phase_txt=ax3d.text2D(0.03,0.96,"",transform=ax3d.transAxes,
                          fontsize=15,fontweight="bold",
                          color=C_LOCK,va="top")
    time_txt =fig.text(0.50,0.018,"",ha="center",
                       fontsize=12,color="#aaaaaa")

    _static_n=[None]
    _dyn_lines=[]          # explicit list of dynamic path line objects
    phase_map={C_REACH:"REACHING",C_SLIDE:"SLIDING",
               C_LOCK:"LOCKED",C_TRANS:"TRANSIENT"}

    def update(frame):
        i=frame
        id_h =id_a[:i+1];  iq_h =iq_a[:i+1]
        rpm_h=rpm_a[:i+1]; t_h  =t_a[:i+1]
        ref_h=ref_a[:i+1]; s_h  =s_a[:i+1]
        cols_h=cols_a[:i+1]; n_pts=len(id_h)

        # Record static collection count on first call
        if _static_n[0] is None:
            _static_n[0]=len(ax3d.collections)

        # Remove dynamic collections (safety)
        while len(ax3d.collections)>_static_n[0]:
            ax3d.collections[-1].remove()

        # Remove previously drawn dynamic path lines explicitly
        for ln in _dyn_lines:
            try: ln.remove()
            except Exception: pass
        _dyn_lines.clear()

        # Draw thick coloured path as phase-segmented lines
        # REACHING = dotted (not yet converged), others = solid
        _PHASE_LS = {C_REACH: (4.0, "dotted"),
                     C_SLIDE: (4.0, "solid"),
                     C_LOCK:  (4.0, "solid"),
                     C_TRANS: (4.0, "solid")}
        if n_pts > 1:
            runs=[]
            run_start=0
            for k in range(1, n_pts):
                if cols_h[k]!=cols_h[k-1]:
                    runs.append((run_start, k, cols_h[run_start]))
                    run_start=k
            runs.append((run_start, n_pts, cols_h[run_start]))
            for rs,re,rc in runs:
                if re>rs:
                    lw, ls = _PHASE_LS.get(rc, (4.0, "solid"))
                    ln, = ax3d.plot(id_h[rs:re], iq_h[rs:re], rpm_h[rs:re],
                                    color=rc, lw=lw, ls=ls, alpha=0.95,
                                    solid_capstyle="round", zorder=7)
                    _dyn_lines.append(ln)

        shadow_line.set_data(id_h,iq_h)
        shadow_line.set_3d_properties(np.zeros(n_pts))
        live_dot.set_data([id_h[-1]],[iq_h[-1]])
        live_dot.set_3d_properties([rpm_h[-1]])
        live_dot.set_color(cols_h[-1])
        drop_line.set_data([id_h[-1],id_h[-1]],[iq_h[-1],iq_h[-1]])
        drop_line.set_3d_properties([0.0,rpm_h[-1]])

        # Speed panel
        sp_live.set_data(t_h,rpm_h)
        sp_dot.set_data([t_h[-1]],[rpm_h[-1]])
        sp_dot.set_color(cols_h[-1])

        # Current panel
        id_live.set_data(t_h,id_h)
        iq_live.set_data(t_h,iq_h)
        id_dot.set_data([t_h[-1]],[id_h[-1]])
        iq_dot.set_data([t_h[-1]],[iq_h[-1]])
        id_dot.set_color(cols_h[-1])
        iq_dot.set_color(cols_h[-1])

        # Tuner cursor: sim_time → iter index
        ti=min(int(t_h[-1]/T_SIM*N_ITER),N_ITER-1)
        tu_cur.set_data([ti,ti],[0,max(tc_costs)])

        # Phase annotation
        cur_col=cols_h[-1]
        lbl=phase_map.get(cur_col,"")
        if cur_col==C_LOCK: lbl=f"LOCKED  {rpm_h[-1]:.0f} RPM"
        phase_txt.set_text(lbl); phase_txt.set_color(cur_col)
        time_txt.set_text(f"t = {t_h[-1]:.3f} s  |  tuner iter {ti}")

        return (live_dot,drop_line,shadow_line,
                sp_live,sp_dot,id_live,iq_live,id_dot,iq_dot,
                tu_cur,phase_txt,time_txt)

    n_frames=len(fx)
    print(f"\n  Building animation ({n_frames} frames) ...")
    ani=animation.FuncAnimation(fig,update,frames=n_frames,
                                interval=55,blit=False)
    print(f"  Saving {gif_path} ...")
    ani.save(gif_path,writer="pillow",fps=20,dpi=120)
    print(f"  Saved  {gif_path}")
    update(n_frames-1)
    fig.savefig(png_path,dpi=170,bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved  {png_path}")


# =============================================================================
# Entry point
# =============================================================================
if __name__=="__main__":
    print("="*65)
    print("  SMC Tune + Animate  —  full pipeline")
    print("="*65)

    # 1. Baseline
    print("\n[1/4] Baseline simulation ...")
    base_gains=dict(SMC_KS_W=_DB42S02.SMC_KS_W,
                    SMC_PHI_W=_DB42S02.SMC_PHI_W,
                    SMC_KS_I=_DB42S02.SMC_KS_I,
                    SMC_PHI_I=_DB42S02.SMC_PHI_I,
                    SMC_LAMBDA_W=_DB42S02.SMC_LAMBDA_W)
    base_data=run_sim(base_gains)
    print(f"  done  final={base_data['rpm'][-1]:.0f} RPM")

    # 2. Tune
    print("\n[2/4] Bayesian GP tuner ...")
    history, best_idx, best_gains = run_tuner()

    # 3. Tuned simulation
    print("\n[3/4] Tuned simulation ...")
    tuned_data=run_sim(best_gains)
    print(f"  done  final={tuned_data['rpm'][-1]:.0f} RPM")

    # 4. Animate
    print("\n[4/4] Building manifold animation ...")
    make_animation(base_data, tuned_data, history, best_idx)

    print("\n[Done]")
    print("  db42s02_smc_tune_and_animate.gif")
    print("  db42s02_smc_tune_and_animate.png")
    print("  smc_tuner_best.py")