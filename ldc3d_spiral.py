"""
SPIRAL-3D : Stream-function -> vector-potential Physics-Informed RAndom-feature Learning for the 3D lid-driven cavity slab + species mixing
===========================================================================================================================================
3D version of the 2D SPIRAL solver, for a Colab T4 (float64 / float32 kernels).  Physics only: no data, no database, no pre-training.

    %run ldc3d_spiral.py                   (Colab:  GPU runtime, T4)

Case (3D_Reference_CFD.pdf / ANSYS Fluent):  50 x 50 x 5 mm slab, lid U = 0.3 m/s on the top (y) wall, left/right walls no-slip,
front/back (z) faces selectable:  Z_WALL = "symmetry" (what the ANSYS case used; the 3D solution is then identical to the 2D one)
                                  Z_WALL = "noslip"   (real 5 mm gap between two walls: truly 3D, w != 0, slower flow)
Passive scalar phi = 1 in a 10 mm circle (centre (35,35) mm) extruded through the depth, injected at t = 4 s (and optionally again later),
dt = 0.001 s like the ANSYS run.  Records what the ANSYS UDFs record:  kinetic_energy.csv, mixing_index.csv, center_of_gravity.csv (X, Y, Z).

Network (what is "learned")
---------------------------
In 3D the stream function does not exist; the divergence-free velocity is the curl of a VECTOR POTENTIAL  A = (A_x, A_y, A_z):

        A_c(x) = sum_k z_{c,k} tanh(w_k . x + b_k)      c = x, y, z     (ONE hidden layer, frozen random w_k, b_k, THREE linear outputs)

A_c is sampled at the mid-points of the staggered grid EDGES of its direction; the velocities on the cell FACES are the exact discrete curl
(u = curl A), so continuity holds identically (to round-off) for every weight vector.  The pressure is removed by the discrete curl of the
momentum equations (taken on the same edges), exactly like the 2D vorticity form.  Residual r(z) = curl( momentum residual ) = 0.
The output weights are found by Newton / Levenberg-Marquardt in kernel (dual) form:     z <- z + Phi^T alpha,
alpha = S^T (S G S^T + lam I)^-1 (-r),   G = Phi Phi^T (block diagonal: one block per component),   S = d r / d A  (sparse),
with a chord (re-used Cholesky) update while the flow changes slowly.  The gauge freedom of A (A + grad chi) makes S G S^T singular along a
known subspace; lam I makes the problem a minimum-norm one and removes it.  Nothing but the Navier-Stokes equations is used.
Scalar: second network phi(x) = sum_k c_k tanh(w_k . x + b_k) at the cell centres, conservative finite-volume advection-diffusion residual
(2nd-order linear upwind like Fluent), BDF2, zero-flux walls, kernel-form solve.  Total phi is conserved to round-off.

Cost (honest)
-------------
The kernel matrix K = S G S^T is DENSE with n_eq = 3 N^2 Nz - ... rows (A has 3 components).  Memory ~ 2 n_eq^2 x (4 or 8) bytes, Cholesky n_eq^3/3 flop.
That is the limit of the method on a 15 GB T4:   N = 30, Nz = 3 (n_eq = 6,003): seconds;   N = 40, Nz = 4 (15,444): float32 factor;
N = 50, Nz = 5 (31,605, the 50 x 50 x 5 mm slab at 1 mm cells): borderline (float32 factor, ~8 GB for K and L).  The ANSYS mesh (0.5 mm, 100 x 100 x 10)
is out of reach for this dense formulation (n_eq ~ 3e5, K = 700 GB).  The script prints the predicted memory before it starts and stops if it cannot fit.
"""
import os, sys, re, time, json, shutil, zipfile, warnings, csv, threading, subprocess, platform, contextlib, resource
import numpy as np
import torch
import scipy.sparse as sps
import matplotlib
import matplotlib.pyplot as plt
from torch.func import vmap, jacrev
from PIL import Image

# ---------------------------------------------------------------- naming + publication style
NAME = "SPIRAL"                         # Stream-function Physics-Informed RAndom-feature Learning (3D: vector potential)
REF = "ANSYS Reference"                 # the ANSYS Fluent csv files
GHIA_LAB = "Ghia et al. (1982)"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.7, "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "figure.dpi": 100, "savefig.bbox": "tight",
})
C_SPIRAL, C_SPIRAL2, C_SPIRAL3, C_REF = "#1f4e9c", "#c0392b", "#2e8b57", "black"
warnings.filterwarnings("ignore", category=UserWarning)


def save_fig(fig, path_no_ext, dpi=300):
    """PNG only (no PDF files)."""
    fig.savefig(path_no_ext + ".png", dpi=dpi)


# =============================================================== settings (edit here or through environment variables)
OUT = os.environ.get("S3_OUT", "/content/ldc3d_results")
QUICK = bool(int(os.environ.get("S3_QUICK", "0")))                  # tiny pipeline check (errors meaningless)
RE_CASE = os.environ.get("S3_RE", "nominal")                         # "nominal" -> Re = 1000 (PDF) | "table" -> 1449 (rho U L / mu of its table)
Z_WALL = os.environ.get("S3_ZWALL", "symmetry")                      # front/back faces: "symmetry" (ANSYS PDF) | "noslip" (5 mm gap, truly 3D)
RCFD_DIR = os.environ.get("RCFD_DIR", "/content/rcfd")               # optional ANSYS csv files (see ANSYS_FILES)
ANSYS_FILES = dict(ke="kinetic_energy.csv", mi="mixing_index.csv", cog="center_of_gravity.csv", uvel="uavgymid.csv", vvel="vavgxmid.csv")   # all optional
ANSYS_COST = json.loads(os.environ.get("S3_ANSYS_COST", "{}"))       # optional, YOUR Fluent numbers, e.g. {"wall_h": 2.5, "cores": 4, "ram_gb": 6, "cpu": "Xeon ..."}

PHYS = dict(L=0.05, U=0.30, rho=1.7, mu=1.76e-5, D=2.5e-5, depth=0.005,          # Table 1 of the PDF
            cx=0.035, cy=0.035, r=0.005)                                         # scalar circle: centre (35, 35) mm, diameter 10 mm
CFG = dict(
    N=30,                   # cells per side in x and y (cell size L/N).  Must be even.  30 -> 1.67 mm (fast), 40 -> 1.25 mm, 50 -> 1 mm (needs ~11 GB and is slow), ANSYS: 0.5 mm
    Nz=None,                # cells through the depth (None -> round(N * depth / L), at least 2: N = 30 -> 3, 40 -> 4, 50 -> 5 = the 50 x 50 x 5 mm slab at 1 mm cells)
    ratio=2.0,              # features per equation (total over the three outputs)
    s_min=2.0, s_max_cells=3.5,           # steepness |w| in [s_min, s_max_cells * N]
    dt=0.005,               # FLOW time step [s].  The spin-up only has to reach the (steady) flat-KE state, so 0.005 is enough; 0.001 = the ANSYS step is 5x slower
    scalar_sub=5,           # scalar steps per flow step (scalar dt = dt / scalar_sub = 0.001 s = the ANSYS step)
    dt_B=0.01,              # scalar time step of the later injection(s) in 'fixed' mode
    save_every_s=0.05,      # the flow history (validation / training plots, scalar run in 'fixed' mode) is stored every this many seconds
    n_start=10,             # backward-Euler sub-steps resolving the impulsive start
    inject="ke_flat",       # "ke_flat": track the kinetic energy and inject phi = 1 (the circle) the moment KE is flat  |  "fixed": inject at t_inj (+ extra_inj) like the ANSYS run (4 s)
    ke_window=1.0, ke_tol=1.0e-3,        # "KE is flat":  |KE(t) - KE(t - window)| / KE(t) < tol
    t_flow_min=4.0, t_flow_max=40.0,     # ke_flat mode: never inject before t_flow_min; if KE is still not flat at t_flow_max inject there anyway
    t_mix=4.0,              # follow the scalar for this long after the injection (ANSYS: 4 s)
    t_inj=4.0, extra_inj=[8.0], t_mix_B=4.0, t_flow=8.0,   # "fixed" mode only: injection times, follow time of the later ones, flow end time (>= t_inj + t_mix)
    ckpt_min=15.0,          # the flow run writes a checkpoint every this many minutes (resume after a Colab disconnect: just run again)
    adv="sou",              # scalar advection: "sou" (2nd-order linear upwind) | "upwind"
    scalar_solver="network",# "network" (kernel-form solve) | "direct" (sparse LU, same solution)
    factor_dtype="auto",    # dtype of the dense kernel K and its Cholesky factor: "float64" | "float32" | "auto" (float32 only if float64 K + L would not fit in 80 % of the GPU memory)
    lam_flow=None, lam_scalar=1e-12,     # None -> 1e-10 (float64) / 1e-6 (float32) of mean(diag K)
    max_it=25, tol_psi=1.0e-8,
    n_frames=120, gif_fps=8, gif_hold_s=1.5, gif_dpi=110, gif_up=3,      # GIF: 120 frames at 8 fps = 15 s (+ 1.5 s pause at both ends); lower gif_fps = slower
    D=None,                 # scalar diffusivity [m2/s]; None -> PDF value 2.5e-5
    seed=1,
)
if QUICK:
    CFG.update(N=12, Nz=2, dt=0.05, dt_B=0.05, save_every_s=0.05, t_flow=8.0, n_frames=16, extra_inj=[], gif_up=1, scalar_solver="direct", ke_tol=2e-2, t_flow_max=20.0)
if os.environ.get("S3_CFG"):
    CFG.update(json.loads(os.environ["S3_CFG"]))
os.makedirs(OUT, exist_ok=True)
if os.environ.get("S3_AGG") or QUICK:
    matplotlib.use("Agg")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
torch.set_default_dtype(DT)
if dev.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
print(f"device: {dev} {torch.cuda.get_device_name(0) if dev.type == 'cuda' else '(CPU: only small meshes are practical)'} | float64 flow state, factor dtype {CFG['factor_dtype']}")
assert Z_WALL in ("symmetry", "noslip"), "S3_ZWALL must be 'symmetry' or 'noslip'"
WALLS = dict(x="noslip", y="noslip", z=Z_WALL)              # y: the top wall is the moving lid


# =============================================================== resource monitor
def _rss_mb():
    try:
        with open("/proc/self/statm") as f: return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except Exception:
        return float("nan")


class Monitor:
    """Samples RAM and (if present) nvidia-smi once per second in a background thread; stage() records wall time, process CPU time,
    peak RAM, peak GPU memory (torch allocator and nvidia-smi) and mean/peak GPU utilisation for a block of work."""

    def __init__(self, period=1.0):
        self.stages = []; self.samples = []; self.period = period; self._stop = threading.Event()
        self.gpu_smi = dev.type == "cuda" and shutil.which("nvidia-smi") is not None
        self.t_start = time.perf_counter(); self.cpu_start = time.process_time()
        self.hw = self._hardware()
        self._th = threading.Thread(target=self._loop, daemon=True); self._th.start()

    @staticmethod
    def _cpu_name():
        try:
            for ln in open("/proc/cpuinfo"):
                if ln.lower().startswith("model name"): return ln.split(":", 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or platform.machine()

    def _hardware(self):
        ram = float("nan")
        try:
            for ln in open("/proc/meminfo"):
                if ln.startswith("MemTotal"): ram = int(ln.split()[1]) / 2 ** 20
        except Exception:
            pass
        try: ncpu = len(os.sched_getaffinity(0))
        except Exception: ncpu = os.cpu_count()
        hw = dict(cpu=self._cpu_name(), cpu_threads=ncpu, ram_gb=ram, os=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                  device=str(dev), dtype="float64")
        if dev.type == "cuda":
            pr = torch.cuda.get_device_properties(0)
            hw.update(gpu=pr.name, gpu_mem_gb=pr.total_memory / 2 ** 30, cuda=torch.version.cuda, gpu_sm=pr.multi_processor_count)
        return hw

    def _query_gpu(self):
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
            u, m, p = [float(x) for x in o.split(",")]; return u, m, p
        except Exception:
            return None

    def _loop(self):
        while not self._stop.wait(self.period):
            s = dict(t=time.perf_counter(), rss=_rss_mb())
            if self.gpu_smi:
                q = self._query_gpu()
                if q: s["gpu_util"], s["gpu_mem"], s["gpu_pow"] = q
            self.samples.append(s)

    @contextlib.contextmanager
    def stage(self, name, **info):
        cuda = dev.type == "cuda"
        if cuda: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        w0 = time.perf_counter(); c0 = time.process_time(); r0 = _rss_mb(); i0 = len(self.samples)
        try:
            yield info
        finally:
            if cuda: torch.cuda.synchronize()
            w = time.perf_counter() - w0; c = time.process_time() - c0; ss = self.samples[i0:]
            rec = dict(stage=name, wall_s=w, cpu_s=c, cpu_cores_busy=c / max(w, 1e-9), ram_peak_mb=float(np.nanmax([r0, _rss_mb()] + [x["rss"] for x in ss])))
            if cuda:
                rec["gpu_torch_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20; rec["gpu_torch_reserved_mb"] = torch.cuda.max_memory_reserved() / 2 ** 20
                ut = [x["gpu_util"] for x in ss if "gpu_util" in x]; gm = [x["gpu_mem"] for x in ss if "gpu_mem" in x]; pw = [x["gpu_pow"] for x in ss if "gpu_pow" in x]
                if ut: rec["gpu_util_mean_pct"] = float(np.mean(ut)); rec["gpu_util_peak_pct"] = float(np.max(ut))
                if gm: rec["gpu_smi_mem_peak_mb"] = float(np.max(gm))
                if pw: rec["gpu_power_mean_w"] = float(np.mean(pw)); rec["gpu_energy_wh"] = float(np.mean(pw)) * w / 3600.0
            rec.update(info); self.stages.append(rec)

    def total(self):
        t = dict(wall_s=time.perf_counter() - self.t_start, cpu_s=time.process_time() - self.cpu_start, ram_peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        if dev.type == "cuda":
            t["gpu_torch_peak_mb"] = max([s.get("gpu_torch_peak_mb", 0) for s in self.stages] or [0])
            gm = [x["gpu_mem"] for x in self.samples if "gpu_mem" in x]
            if gm: t["gpu_smi_mem_peak_mb"] = float(max(gm))
        return t

MON = Monitor()
L_, U_, RHO, DEPTH = PHYS["L"], PHYS["U"], PHYS["rho"], PHYS["depth"]
T_CONV = L_ / U_                                   # convective time L/U [s]
RE_TABLE = RHO * U_ * L_ / PHYS["mu"]
ZR = DEPTH / L_                                    # depth / L  (dimensionless slab depth)
KE_SCALE = RHO * U_ ** 2 * L_ ** 3                 # J per unit of the dimensionless  sum 0.5 |u|^2 dV   (dV in units of L^3)


# =============================================================== staggered 3D grid, vector-potential curl, ghost cells
_B3 = {}


def grid_sizes(N, Nz):
    Nx = Ny = N
    nUx, nUy, nUz = (Nx - 1) * Ny * Nz, Nx * (Ny - 1) * Nz, Nx * Ny * (Nz - 1)
    nAx, nAy, nAz = Nx * (Ny - 1) * (Nz - 1), (Nx - 1) * Ny * (Nz - 1), (Nx - 1) * (Ny - 1) * Nz
    return dict(nUx=nUx, nUy=nUy, nUz=nUz, nU=nUx + nUy + nUz, nAx=nAx, nAy=nAy, nAz=nAz, nA=nAx + nAy + nAz)


def build3(N, Nz, walls):
    """Index maps, ghost-cell reflections and sparse curl operators of the staggered grid (cells N x N x Nz, h = L/N in x,y; hz = depth/Nz).
    faces : u_x(i, j+1/2, k+1/2) i=1..N-1 ; u_y(i+1/2, j, k+1/2) j=1..N-1 ; u_z(i+1/2, j+1/2, k) k=1..Nz-1   (interior faces; wall faces are 0)
    edges : A_x(i+1/2, j, k) j=1..N-1,k=1..Nz-1 ; A_y(i, j+1/2, k) ; A_z(i, j, k+1/2)   (tangential A = 0 on the boundary -> wall-normal u = 0)
    u = curl A  (exactly divergence free).   Wall types per axis: 'noslip' (ghost = -u, lid: -u + 2 U_lid) or 'symmetry' (ghost = +u)."""
    key = (N, Nz, tuple(sorted(walls.items())))
    if key in _B3:
        return _B3[key]
    Nx = Ny = N; hx = hy = 1.0 / N; hz = ZR / Nz; sz_ = grid_sizes(N, Nz)
    sx, sy, sz = [(-1.0 if walls[a] == "noslip" else 1.0) for a in "xyz"]; lid_on = walls["y"] == "noslip"
    nUx, nUy, nUz, nU = sz_["nUx"], sz_["nUy"], sz_["nUz"], sz_["nU"]
    nAx, nAy, nAz, nA = sz_["nAx"], sz_["nAy"], sz_["nAz"], sz_["nA"]
    offUy, offUz, offAy, offAz = nUx, nUx + nUy, nAx, nAx + nAy
    bc = lambda I, J, K: np.broadcast_arrays(np.asarray(I), np.asarray(J), np.asarray(K))

    def fin(idx, sg, cs, wall):
        return idx, np.where(wall, 0.0, sg), np.where(wall, 0.0, cs)

    def Ux(I, J, K):
        I, J, K = bc(I, J, K); wall = (I <= 0) | (I >= Nx)
        idx = ((np.clip(I, 1, Nx - 1) - 1) * Ny + np.clip(J, 0, Ny - 1)) * Nz + np.clip(K, 0, Nz - 1)
        sg = np.where((J < 0) | (J >= Ny), sy, 1.0) * np.where((K < 0) | (K >= Nz), sz, 1.0)
        cs = np.where((J >= Ny) & lid_on, 2.0, 0.0)                                  # lid: ghost = -u + 2 * 1
        return fin(idx, sg, cs, wall)

    def Uy(I, J, K):
        I, J, K = bc(I, J, K); wall = (J <= 0) | (J >= Ny)
        idx = offUy + (np.clip(I, 0, Nx - 1) * (Ny - 1) + (np.clip(J, 1, Ny - 1) - 1)) * Nz + np.clip(K, 0, Nz - 1)
        sg = np.where((I < 0) | (I >= Nx), sx, 1.0) * np.where((K < 0) | (K >= Nz), sz, 1.0)
        return fin(idx, sg, np.zeros(idx.shape), wall)

    def Uz(I, J, K):
        I, J, K = bc(I, J, K); wall = (K <= 0) | (K >= Nz)
        idx = offUz + (np.clip(I, 0, Nx - 1) * Ny + np.clip(J, 0, Ny - 1)) * (Nz - 1) + (np.clip(K, 1, Nz - 1) - 1)
        sg = np.where((I < 0) | (I >= Nx), sx, 1.0) * np.where((J < 0) | (J >= Ny), sy, 1.0)
        return fin(idx, sg, np.zeros(idx.shape), wall)

    mg = lambda a, b, c: [x.ravel() for x in np.meshgrid(a, b, c, indexing="ij")]
    st7 = lambda F, I, J, K: [F(I, J, K), F(I + 1, J, K), F(I - 1, J, K), F(I, J + 1, K), F(I, J - 1, K), F(I, J, K + 1), F(I, J, K - 1)]
    I, J, K = mg(np.arange(1, Nx), np.arange(Ny), np.arange(Nz))                       # x-faces
    sx_ = dict(s=st7(Ux, I, J, K), a=[Uy(I - 1, J, K), Uy(I, J, K), Uy(I - 1, J + 1, K), Uy(I, J + 1, K)],
               b=[Uz(I - 1, J, K), Uz(I, J, K), Uz(I - 1, J, K + 1), Uz(I, J, K + 1)])
    I, J, K = mg(np.arange(Nx), np.arange(1, Ny), np.arange(Nz))                       # y-faces
    sy_ = dict(s=st7(Uy, I, J, K), a=[Ux(I, J - 1, K), Ux(I + 1, J - 1, K), Ux(I, J, K), Ux(I + 1, J, K)],
               b=[Uz(I, J - 1, K), Uz(I, J, K), Uz(I, J - 1, K + 1), Uz(I, J, K + 1)])
    I, J, K = mg(np.arange(Nx), np.arange(Ny), np.arange(1, Nz))                       # z-faces
    sz__ = dict(s=st7(Uz, I, J, K), a=[Ux(I, J, K - 1), Ux(I + 1, J, K - 1), Ux(I, J, K), Ux(I + 1, J, K)],
                b=[Uy(I, J, K - 1), Uy(I, J + 1, K - 1), Uy(I, J, K), Uy(I, J + 1, K)])

    # ---- vector potential edges and the curl  (faces <- edges)
    def Ax(I, J, K):
        I, J, K = bc(I, J, K); ok = (J >= 1) & (J <= Ny - 1) & (K >= 1) & (K <= Nz - 1) & (I >= 0) & (I <= Nx - 1)
        return (np.clip(I, 0, Nx - 1) * (Ny - 1) + (np.clip(J, 1, Ny - 1) - 1)) * (Nz - 1) + (np.clip(K, 1, Nz - 1) - 1), ok

    def Ay(I, J, K):
        I, J, K = bc(I, J, K); ok = (I >= 1) & (I <= Nx - 1) & (K >= 1) & (K <= Nz - 1) & (J >= 0) & (J <= Ny - 1)
        return offAy + ((np.clip(I, 1, Nx - 1) - 1) * Ny + np.clip(J, 0, Ny - 1)) * (Nz - 1) + (np.clip(K, 1, Nz - 1) - 1), ok

    def Az(I, J, K):
        I, J, K = bc(I, J, K); ok = (I >= 1) & (I <= Nx - 1) & (J >= 1) & (J <= Ny - 1) & (K >= 0) & (K <= Nz - 1)
        return offAz + ((np.clip(I, 1, Nx - 1) - 1) * (Ny - 1) + (np.clip(J, 1, Ny - 1) - 1)) * Nz + np.clip(K, 0, Nz - 1), ok
    rr, cc, vv = [], [], []

    def add(row, EF, I, J, K, val):
        c, ok = EF(I, J, K); rr.append(row[ok]); cc.append(c[ok]); vv.append(np.full(ok.sum(), val))
    I, J, K = mg(np.arange(1, Nx), np.arange(Ny), np.arange(Nz)); row = ((I - 1) * Ny + J) * Nz + K            # u_x = dy Az - dz Ay
    add(row, Az, I, J + 1, K, 1 / hy); add(row, Az, I, J, K, -1 / hy); add(row, Ay, I, J, K + 1, -1 / hz); add(row, Ay, I, J, K, 1 / hz)
    I, J, K = mg(np.arange(Nx), np.arange(1, Ny), np.arange(Nz)); row = offUy + (I * (Ny - 1) + (J - 1)) * Nz + K   # u_y = dz Ax - dx Az
    add(row, Ax, I, J, K + 1, 1 / hz); add(row, Ax, I, J, K, -1 / hz); add(row, Az, I + 1, J, K, -1 / hx); add(row, Az, I, J, K, 1 / hx)
    I, J, K = mg(np.arange(Nx), np.arange(Ny), np.arange(1, Nz)); row = offUz + (I * Ny + J) * (Nz - 1) + (K - 1)   # u_z = dx Ay - dy Ax
    add(row, Ay, I + 1, J, K, 1 / hx); add(row, Ay, I, J, K, -1 / hx); add(row, Ax, I, J + 1, K, -1 / hy); add(row, Ax, I, J, K, 1 / hy)
    Cur = sps.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(nU, nA)).tocsr()
    # ---- edge positions (feature inputs, dimensionless: x, y in [0,1], z in [0, depth/L])
    I, J, K = mg(np.arange(Nx), np.arange(1, Ny), np.arange(1, Nz)); pAx = np.stack([(I + .5) * hx, J * hy, K * hz], 1)
    I, J, K = mg(np.arange(1, Nx), np.arange(Ny), np.arange(1, Nz)); pAy = np.stack([I * hx, (J + .5) * hy, K * hz], 1)
    I, J, K = mg(np.arange(1, Nx), np.arange(1, Ny), np.arange(Nz)); pAz = np.stack([I * hx, J * hy, (K + .5) * hz], 1)
    tt = lambda sl: [(torch.as_tensor(i_, device=dev), torch.as_tensor(s_, device=dev, dtype=DT), torch.as_tensor(c, device=dev, dtype=DT)) for i_, s_, c in sl]
    _B3[key] = dict(N=N, Nz=Nz, walls=walls, h=(hx, hy, hz), **sz_, sten=dict(x=sx_, y=sy_, z=sz__), Cur=Cur, CurT=Cur.T.tocsr(),
                    sten_t={d: {k: tt(v) for k, v in s.items()} for d, s in (("x", sx_), ("y", sy_), ("z", sz__))},
                    pts=[torch.as_tensor(p, device=dev, dtype=DT) for p in (pAx, pAy, pAz)], sizes=(nAx, nAy, nAz))
    return _B3[key]


def _g(sl, o):
    return torch.stack([sg * o[idx] + cs for idx, sg, cs in sl], 1)


def _fns3(B, Re):
    hx, hy, hz = B["h"]

    def mk(d, o1, o2):
        def f(s7, a, b):
            c, e, w, n, so, t, bt = s7; vel = [None, None, None]; vel[d] = c; vel[o1] = a.mean(); vel[o2] = b.mean()
            return (vel[0] * (e - w) / (2 * hx) + vel[1] * (n - so) / (2 * hy) + vel[2] * (t - bt) / (2 * hz)
                    - ((e + w - 2 * c) / hx ** 2 + (n + so - 2 * c) / hy ** 2 + (t + bt - 2 * c) / hz ** 2) / Re)
        return f
    return dict(x=mk(0, 1, 2), y=mk(1, 0, 2), z=mk(2, 0, 1))


def momentum3(o, B, Re, tm=0.0, hc=None):
    """Momentum residual R at every interior face (central convection with face-averaged transport velocities, 2nd-order Laplacian),
    plus the time term (tm * o + hc)."""
    f = _fns3(B, Re); out = []
    for d in "xyz":
        st = B["sten_t"][d]; out.append(vmap(f[d])(_g(st["s"], o), _g(st["a"], o), _g(st["b"], o)))
    R = torch.cat(out)
    return R + tm * o + hc if tm else R


def _t(M, dtype=None):
    return torch.sparse_csr_tensor(torch.as_tensor(M.indptr, dtype=torch.int64, device=dev), torch.as_tensor(M.indices, dtype=torch.int64, device=dev),
                                   torch.as_tensor(M.data, device=dev, dtype=dtype or DT), size=M.shape)


def _spmv(Mt, v):
    return torch.sparse.mm(Mt, v[:, None])[:, 0]


def jac3(o, B, Re, tm=0.0):
    """S = d r / d A  (sparse, scipy) with r = Cur^T (momentum residual): chain  Cur^T (J + tm I) Cur  with J from vmap(jacrev) of the local stencils."""
    f = _fns3(B, Re); nU = B["nU"]; offs = dict(x=0, y=B["nUx"], z=B["nUx"] + B["nUy"]); comp = dict(x=("x", "y", "z"), y=("y", "x", "z"), z=("z", "x", "y"))
    rr, cc, vv = [], [], []
    for d in "xyz":
        st = B["sten"][d]; stt = B["sten_t"][d]
        Js = vmap(jacrev(f[d], argnums=(0, 1, 2)))(_g(stt["s"], o), _g(stt["a"], o), _g(stt["b"], o))
        m = Js[0].shape[0]; row0 = offs[d]
        for part, Jm, cd in zip(("s", "a", "b"), Js, comp[d]):
            Jm = Jm.cpu().numpy()
            for k, (idx, sg, cs) in enumerate(st[part]):
                rr.append(row0 + np.arange(m)); cc.append(idx); vv.append(Jm[:, k] * sg)
    J = sps.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(nU, nU)).tocsr()
    if tm:
        J = J + tm * sps.identity(nU, format="csr")
    return (B["CurT"] @ (J @ B["Cur"])).tocsr()


# =============================================================== hidden layer (frozen random tanh features, shared by the three outputs)
def make_features3(n_feat, s_min, s_max, seed):
    g = torch.Generator().manual_seed(seed)
    s = torch.exp(torch.empty(n_feat).uniform_(np.log(s_min), np.log(s_max), generator=g))
    d = torch.randn(n_feat, 3, generator=g); d = d / d.norm(dim=1, keepdim=True)
    x0 = torch.rand(n_feat, 3, generator=g) * torch.tensor([1.0, 1.0, ZR])
    w = s[:, None] * d
    return w.T.contiguous().to(dev), (-(x0 * w).sum(1)).to(dev)                 # W (3,P), b (P)


def kernel_blocks(pts_list, W, b, chunk=4096):
    """G_c = Phi_c Phi_c^T for every output c  (Phi_c = tanh(pts_c W + b)); Phi is never stored (chunked over the features)."""
    Gs = []
    for pts in pts_list:
        n = pts.shape[0]; G = torch.zeros(n, n, device=dev, dtype=DT)
        for p0 in range(0, W.shape[1], chunk):
            Ph = torch.tanh(pts @ W[:, p0:p0 + chunk] + b[p0:p0 + chunk]); G += Ph @ Ph.T
        Gs.append(G)
    return Gs


TEMP_BYTES = 1.2e9          # assembly buffers (chunks of S G, S^T products) + cuSOLVER workspace


def pick_factor_dtype(N, Nz, cfg):
    """float64 unless G + K (factorised IN PLACE, K -> L) would not fit in 85 % of the GPU memory (or n_eq > 14000 on a CPU): then float32 for K / L
    (the Newton / chord iteration still uses float64 residuals, so the converged solution is the same; it needs more iterations)."""
    fd = cfg["factor_dtype"]
    if fd in ("float32", "float64"):
        return torch.float32 if fd == "float32" else torch.float64
    s = grid_sizes(N, Nz); need64 = 8 * (s["nAx"] ** 2 + s["nAy"] ** 2 + s["nAz"] ** 2) + 8 * s["nA"] ** 2 + TEMP_BYTES
    cap = 0.85 * torch.cuda.get_device_properties(0).total_memory if dev.type == "cuda" else 14000 ** 2 * 8 * 2.6
    return torch.float64 if need64 <= cap else torch.float32


def predict_memory(N, Nz, cfg):
    """GPU memory model [bytes] of the flow network and of the scalar network (they are never alive together at the peak)."""
    s = grid_sizes(N, Nz); nA = s["nA"]; fd = pick_factor_dtype(N, Nz, cfg); b = 4 if fd == torch.float32 else 8
    G = 8 * (s["nAx"] ** 2 + s["nAy"] ** 2 + s["nAz"] ** 2); KL = b * nA ** 2; Mc = N * N * Nz
    return dict(nA=nA, factor_dtype=str(fd).replace("torch.", ""), G=G, K_L=KL, flow=G + KL + TEMP_BYTES, scalar=2 * 8 * Mc ** 2 + 8 * Mc ** 2 + 0.3e9, cells=Mc,
                chol_flop=nA ** 3 / 3, features=int(cfg["ratio"] * nA), nU=s["nU"])


def sym_colmajor(n, dtype):
    """Uninitialised n x n matrix with column-major strides, so that LAPACK/cuSOLVER can factorise it IN PLACE (no second copy)."""
    return torch.empty(n, n, device=dev, dtype=dtype).mT


def chol_inplace(K):
    """In-place Cholesky of the column-major symmetric matrix K (K becomes L, lower).  Returns info (0 = success; K is destroyed otherwise)."""
    info = torch.empty((), dtype=torch.int32, device=K.device)
    torch.linalg.cholesky_ex(K, out=(K, info))
    return int(info.item())


def chol_solve(L, b):
    """Solve (L L^T) x = b with two triangular solves that read L in place (cholesky_solve would clone L on every call)."""
    y = torch.linalg.solve_triangular(L, b[:, None], upper=False)
    return torch.linalg.solve_triangular(L.mT, y, upper=True)[:, 0]


class FlowNet3:
    """A(x) = sum_k z_k tanh(w_k.x + b_k) (3 outputs).  State = beta with  A = G beta  (z = Phi^T beta; Phi is never stored).
    Kernel-form Levenberg-Marquardt / chord-Newton time stepping of  curl(momentum) = 0."""

    def __init__(self, N, Nz, walls, Re, cfg):
        self.N, self.Nz, self.Re, self.cfg = N, Nz, float(Re), cfg
        B = self.B = build3(N, Nz, walls); self.nA = B["nA"]; self.sl = np.cumsum([0] + list(B["sizes"]))
        self.P = int(cfg["ratio"] * self.nA / 3)                                                # features; total weights = 3 P
        self.fd = pick_factor_dtype(N, Nz, cfg); self.lam0 = cfg["lam_flow"] or (1e-10 if self.fd == torch.float64 else 1e-6)
        W, b = make_features3(self.P, cfg["s_min"], cfg["s_max_cells"] * N, cfg["seed"]); self.W, self.b = W, b
        t = time.time(); self.G = kernel_blocks(B["pts"], W, b)
        self.beta = torch.zeros(self.nA, device=dev, dtype=DT); self.beta_m = self.beta.clone()
        self.Cur_t = _t(B["Cur"]); self.CurT_t = _t(B["CurT"])
        self.fact = None; self.n_fact = 0; self.n_iter = 0; self._stale = True; self.fact_tm = 0.0
        print(f"flow network: {self.P:,} tanh features x 3 outputs ({3 * self.P:,} weights) -> {self.nA:,} edge equations, kernel blocks built in {time.time() - t:.1f} s; "
              f"factor dtype {str(self.fd).replace('torch.', '')}", flush=True)

    def Gmul(self, v):
        return torch.cat([self.G[c] @ v[self.sl[c]:self.sl[c + 1]] for c in range(3)])

    def A(self, beta=None):
        return self.Gmul(self.beta if beta is None else beta)

    def faces(self, A=None):
        return _spmv(self.Cur_t, self.A() if A is None else A)

    def residual(self, A, tm, hc):
        return _spmv(self.CurT_t, momentum3(_spmv(self.Cur_t, A), self.B, self.Re, tm, hc))

    def weights(self):
        """The output weights z (3, P) of the network (z_c = Phi_c^T beta_c), computed on demand."""
        Z = torch.zeros(3, self.P, device=dev, dtype=DT)
        for c in range(3):
            for p0 in range(0, self.P, 4096):
                Z[c, p0:p0 + 4096] = torch.tanh(self.B["pts"][c] @ self.W[:, p0:p0 + 4096] + self.b[p0:p0 + 4096]).T @ self.beta[self.sl[c]:self.sl[c + 1]]
        return Z

    def _factor(self, A, tm, lam):
        """K = S G S^T (+ lam scale I), Cholesky IN PLACE (K -> L).  Fails over to a float32 factor if the GPU runs out of memory."""
        self.fact = None
        if dev.type == "cuda": torch.cuda.empty_cache()
        try:
            return self._factor_dt(A, tm, lam)
        except torch.cuda.OutOfMemoryError:
            if self.fd == torch.float32: raise
            print("   !!! GPU memory exhausted by the float64 factor: switching to a float32 factor (more Newton iterations per step, same solution)", flush=True)
            self.fact = None; self.fd = torch.float32; self.lam0 = max(self.lam0, 1e-6); lam = max(lam, 1e-6); torch.cuda.empty_cache()
            return self._factor_dt(A, tm, lam)

    def _factor_dt(self, A, tm, lam):
        B = self.B; fd = self.fd; nA = self.nA; S = jac3(self.faces(A), B, self.Re, tm); St_t = _t(S.T.tocsr()); Scsc = S.tocsc()
        Scs = [Scsc[:, self.sl[c]:self.sl[c + 1]].tocsr() for c in range(3)]; ch = 1024; scale = None; L = None

        def assemble():
            K = sym_colmajor(nA, fd)
            for c in range(3):
                Sc = Scs[c]; Sc_t = _t(Sc, fd); Gc = self.G[c] if fd == DT else self.G[c].to(fd)
                for r0 in range(0, nA, ch):
                    Bb = torch.sparse.mm(_t(Sc[r0:r0 + ch], fd), Gc)
                    part = torch.sparse.mm(Sc_t, Bb.T.contiguous()); del Bb
                    if c == 0: K[:, r0:r0 + ch] = part
                    else: K[:, r0:r0 + ch] += part
                    del part
                del Sc_t
                if fd != DT: del Gc
            return K
        for attempt in range(14):
            K = assemble()
            if scale is None: scale = K.diagonal().mean().item()
            K.diagonal().add_(lam * scale)
            if chol_inplace(K) == 0:
                L = K; break
            del K                                                           # destroyed by the failed factorisation: rebuild with a larger lam
            lam *= 10
        if L is None: raise RuntimeError("Cholesky of the kernel matrix failed for every lam tried")
        self.fact = (St_t, L); self.n_fact += 1
        return lam

    def step(self, tm, hc, extrapolate=True):
        """Solve for the next time level.  tm = a0/dt*, hc = history/dt* (flat face vector)."""
        cfg = self.cfg
        bg = (2 * self.beta - self.beta_m) if extrapolate else self.beta.clone()
        A0 = self.Gmul(bg); alpha = torch.zeros(self.nA, device=dev, dtype=DT); Acur = A0
        lam = self.lam0; r = self.residual(A0, tm, hc); f = (r ** 2).mean().item()
        fresh = False; last_rel = np.inf; used = 0
        tr = dict(f0=f, fs=[f], rel=[np.nan], nf0=self.n_fact)
        for it in range(cfg["max_it"]):
            if self.fact is None or (it == 0 and (self._stale or abs(tm - self.fact_tm) > 0.05 * tm)):
                lam = self._factor(Acur, tm, lam); fresh = True; self.fact_tm = tm
            St_t, L = self.fact
            y = chol_solve(L, (-r).to(self.fd)).to(DT)
            da = _spmv(St_t, y)
            Gda = self.Gmul(da); An = Acur + Gda
            rn = self.residual(An, tm, hc); fn = (rn ** 2).mean().item()
            if np.isfinite(fn) and fn < f:
                dps = Gda.abs().max().item(); pm = An.abs().max().item()
                if fn > 0.5 * f and not fresh and last_rel > cfg["tol_psi"]:
                    self.fact = None                                       # chord contracts too slowly: rebuild at the new iterate
                alpha = alpha + da; Acur, r, f = An, rn, fn; self.n_iter += 1; fresh = False; used += 1; last_rel = dps / max(pm, 1e-12)
                tr["fs"].append(fn); tr["rel"].append(last_rel)
                if last_rel < cfg["tol_psi"] or f < 1e-26:
                    break
            else:
                if last_rel < 1e-6:
                    break
                if fresh:
                    lam *= 10; self.fact = None; fresh = False
                    if lam > 1e3: break
                else:
                    self.fact = None
        tr["n_it"] = used; tr["refact"] = self.n_fact - tr.pop("nf0"); self.last = tr
        self._stale = used > 4
        self.beta_m = self.beta
        self.beta = bg + alpha
        return f


# =============================================================== diagnostics on the face vector
def split_faces(o, N, Nz):
    s = grid_sizes(N, Nz)
    return (o[:s["nUx"]].reshape(N - 1, N, Nz), o[s["nUx"]:s["nUx"] + s["nUy"]].reshape(N, N - 1, Nz), o[s["nUx"] + s["nUy"]:].reshape(N, N, Nz - 1))


def divergence(o, N, Nz):
    """Discrete divergence of the face velocities in every cell (zero normal velocity on the walls)."""
    ux, uy, uz = split_faces(o, N, Nz); hz = ZR / Nz; h = 1.0 / N
    px = torch.nn.functional.pad(ux, (0, 0, 0, 0, 1, 1)); py = torch.nn.functional.pad(uy, (0, 0, 1, 1)); pz = torch.nn.functional.pad(uz, (1, 1))
    return (px[1:] - px[:-1]) / h + (py[:, 1:] - py[:, :-1]) / h + (pz[:, :, 1:] - pz[:, :, :-1]) / hz


def cell_velocity(o, N, Nz):
    """Cell-centre velocity components (dimensionless), shape (N, N, Nz) each (works for torch and numpy via the same slicing)."""
    ux, uy, uz = split_faces(o, N, Nz)
    px = torch.nn.functional.pad(ux, (0, 0, 0, 0, 1, 1)); py = torch.nn.functional.pad(uy, (0, 0, 1, 1)); pz = torch.nn.functional.pad(uz, (1, 1))
    return 0.5 * (px[1:] + px[:-1]), 0.5 * (py[:, 1:] + py[:, :-1]), 0.5 * (pz[:, :, 1:] + pz[:, :, :-1])


def kinetic_energy(o, N, Nz):
    """0.5 sum |u_c|^2 dV  (dimensionless, dV in L^3)."""
    uc, vc, wc = cell_velocity(o, N, Nz)
    return 0.5 * float((uc ** 2 + vc ** 2 + wc ** 2).sum()) * (1.0 / N ** 2) * (ZR / Nz)


def run_flow(N, Nz, Re, cfg, walls, ckpt_path=None, key=None):
    """Spin-up from rest.  cfg['inject'] == 'ke_flat': run until the kinetic energy is flat (|KE(t)-KE(t-window)|/KE(t) < ke_tol, not before t_flow_min,
    give up at t_flow_max) and stop there - that time is the injection time.  cfg['inject'] == 'fixed': run to t_flow and only report when KE became flat.
    A checkpoint (state + histories) is written every ckpt_min minutes to ckpt_path and used to resume an interrupted run (same configuration)."""
    fn = FlowNet3(N, Nz, walls, Re, cfg); dt_s = cfg["dt"]; dts = dt_s / T_CONV; B = fn.B; nU = B["nU"]
    stride = max(1, int(round(cfg["save_every_s"] / dt_s))); fixed = cfg["inject"] == "fixed"
    zero = torch.zeros(nU, device=dev, dtype=DT)
    ts, KE, O, tO = [0.0], [0.0], [zero.cpu().numpy()], [0.0]
    TR = dict(t=[], f0=[], f=[], n_it=[], refact=[], rel=[]); TRACES = {}
    maxdiv = 0.0; plateau = None; elapsed0 = 0.0

    def _rec(t_, n_):
        l_ = fn.last; TR["t"].append(t_); TR["f0"].append(l_["f0"]); TR["f"].append(l_["fs"][-1]); TR["n_it"].append(l_["n_it"]); TR["refact"].append(l_["refact"]); TR["rel"].append(l_["rel"][-1])
        if n_ <= 5 or n_ % 250 == 0: TRACES[str(n_)] = dict(t=t_, f=[float(x) for x in l_["fs"]], rel=[float(x) for x in l_["rel"]])
    ck = None
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, weights_only=False)
            if ck["key"] != key: ck = None
        except Exception as e:
            print("checkpoint not usable:", e, flush=True); ck = None
    t0 = time.time()
    if ck is not None:
        ts, KE, O, tO, TR, TRACES, n, plateau, maxdiv, elapsed0 = [ck[k] for k in ("ts", "KE", "O", "tO", "TR", "TRACES", "n", "plateau", "maxdiv", "elapsed")]
        fn.beta = ck["beta"].to(dev); fn.beta_m = ck["beta_m"].to(dev); fn.n_iter = ck["n_iter"]; fn.n_fact = ck["n_fact"]
        o_n = ck["o_n"].to(dev); o_nm1 = ck["o_nm1"].to(dev); fn._stale = True
        print(f"   >>> resumed from checkpoint at step {n} (t = {n * dt_s:.2f} s, {elapsed0 / 60:.1f} min of earlier work)", flush=True)
    else:
        o_prev = zero.clone(); nsub = cfg["n_start"]
        for k in range(nsub):                                                    # first interval [0, dt]: backward-Euler sub-steps (impulsive start)
            tm = 1.0 / (dts / nsub); hc = -o_prev / (dts / nsub)
            fn.step(tm, hc, extrapolate=(k > 0)); o_prev = fn.faces(); _rec((k + 1) * dt_s / nsub, 0)
        o_n = o_prev.clone(); o_nm1 = zero.clone(); fn.beta_m = torch.zeros_like(fn.beta)       # level n-1 is the rest state
        ts.append(dt_s); KE.append(kinetic_energy(o_n, N, Nz) * KE_SCALE)
        if stride == 1: O.append(o_n.cpu().numpy()); tO.append(dt_s)
        n = 1
    nmax = int(round((cfg["t_flow"] if fixed else cfg["t_flow_max"]) / dt_s)); nw = int(round(cfg["ke_window"] / dt_s)); last_ck = time.time()
    ck_steps = cfg.get("_ckpt_steps")
    print(f"{'t [s]':>7} {'KE [J]':>12} {'dKE/KE per window':>18} {'|A|max':>9} {'Newton its':>10} {'factors':>8} {'wall [min]':>10}", flush=True)
    while n < nmax and not (plateau is not None and not fixed):
        tm = 1.5 / dts; hc = (-2.0 * o_n + 0.5 * o_nm1) / dts
        fn.step(tm, hc)
        o_nm1, o_n = o_n, fn.faces(); n += 1; t = n * dt_s; _rec(t, n); ts.append(t); KE.append(kinetic_energy(o_n, N, Nz) * KE_SCALE)
        if n % stride == 0: O.append(o_n.cpu().numpy()); tO.append(t)
        rel = abs(KE[-1] - KE[-1 - nw]) / KE[-1] if n > nw else np.nan
        if plateau is None and t >= cfg["t_flow_min"] - 1e-9 and np.isfinite(rel) and rel < cfg["ke_tol"]:
            plateau = t; print(f"   >>> kinetic energy is flat at t = {t:.2f} s  (window change {rel:.2e} < {cfg['ke_tol']:g})" + ("" if fixed else "  -> injecting the scalar here"), flush=True)
        if n % max(1, int(round(0.5 / dt_s))) == 0:
            md = divergence(o_n, N, Nz).abs().max().item(); maxdiv = max(maxdiv, md)
            print(f"{t:7.2f} {KE[-1]:12.5e} {rel:18.3e} {fn.A().abs().max().item():9.5f} {fn.n_iter:10d} {fn.n_fact:8d} {(elapsed0 + time.time() - t0) / 60:10.1f}", flush=True)
        due = (ck_steps and n % ck_steps == 0) or (not ck_steps and ckpt_path and time.time() - last_ck > 60 * cfg["ckpt_min"])
        if ckpt_path and due:
            torch.save(dict(key=key, ts=ts, KE=KE, O=O, tO=tO, TR=TR, TRACES=TRACES, n=n, plateau=plateau, maxdiv=maxdiv, elapsed=elapsed0 + time.time() - t0, beta=fn.beta.cpu(), beta_m=fn.beta_m.cpu(),
                            n_iter=fn.n_iter, n_fact=fn.n_fact, o_n=o_n.cpu(), o_nm1=o_nm1.cpu()), ckpt_path + ".tmp"); os.replace(ckpt_path + ".tmp", ckpt_path); last_ck = time.time()
    found = plateau is not None
    if not found:
        print(f"   (KE has not reached the flatness criterion by t = {n * dt_s:.2f} s; the scalar is injected into the last computed flow)", flush=True)
        plateau = ts[-1]
    if tO[-1] < ts[-1] - 1e-12: O.append(o_n.cpu().numpy()); tO.append(ts[-1])          # the last level is always stored (the scalar sees a frozen flow after it)
    md = divergence(o_n, N, Nz).abs().max().item(); maxdiv = max(maxdiv, md)
    print(f"   max |div u| over the run (sampled): {maxdiv:.2e}  (dimensionless; round-off level => continuity holds identically)", flush=True)
    if ckpt_path and os.path.exists(ckpt_path): os.remove(ckpt_path)
    return dict(t=np.array(ts), KE=np.array(KE), O=np.array(O), t_psi=np.array(tO), t_plateau=plateau, plateau_found=found, fn=fn, maxdiv=maxdiv, N=N, Nz=Nz,
                stats=dict(features=int(3 * fn.P), equations=int(fn.nA), newton_its=int(fn.n_iter), factorisations=int(fn.n_fact), steps=int(n)),
                train=dict(arrays={k_: np.array(v_, float) for k_, v_ in TR.items()}, traces=TRACES))


# =============================================================== scalar: phi network + 3D finite-volume transport operator
def scalar_operator3(ux, uy, uz, N, Nz, D, scheme):
    """A phi = div(u phi) - D lap phi  (conservative FV, zero-flux walls) as a sparse (M x M) matrix, cell (i,j,k) -> (i*N + j)*Nz + k.
    ux (N-1,N,Nz), uy (N,N-1,Nz), uz (N,N,Nz-1): interior face velocities (dimensionless).  Linear-upwind (2nd order) or 1st-order upwind."""
    hx = hy = 1.0 / N; hz = ZR / Nz; sou = (scheme == "sou")
    rows, cols, vals = [], [], []
    ix = lambda i, j, k: (i * N + j) * Nz + k

    def faceset(q, Wc, Ec, Wc2, Ec2, ok_w2, ok_e2, h):
        pos = q >= 0
        c1 = np.where(pos, Wc, Ec); use2 = np.where(pos, ok_w2, ok_e2) & sou; c2 = np.where(pos, Wc2, Ec2)
        w1 = np.where(use2, 1.5, 1.0); w2 = np.where(use2, -0.5, 0.0)
        for cell, sgn in ((Wc, +1.0), (Ec, -1.0)):                         # + flux leaves W, enters E
            rows.append(cell); cols.append(c1); vals.append(sgn * q * w1 / h)
            rows.append(cell); cols.append(np.where(use2, c2, c1)); vals.append(sgn * q * w2 / h)
        d = D / h ** 2
        rows.extend([Wc, Wc, Ec, Ec]); cols.extend([Wc, Ec, Ec, Wc]); vals.extend([np.full_like(q, d), np.full_like(q, -d), np.full_like(q, d), np.full_like(q, -d)])
    mg = lambda a, b, c: [x.ravel() for x in np.meshgrid(a, b, c, indexing="ij")]
    I, J, K = mg(np.arange(1, N), np.arange(N), np.arange(Nz))
    faceset(ux[I - 1, J, K], ix(I - 1, J, K), ix(I, J, K), ix(np.maximum(I - 2, 0), J, K), ix(np.minimum(I + 1, N - 1), J, K), I - 2 >= 0, I + 1 <= N - 1, hx)
    I, J, K = mg(np.arange(N), np.arange(1, N), np.arange(Nz))
    faceset(uy[I, J - 1, K], ix(I, J - 1, K), ix(I, J, K), ix(I, np.maximum(J - 2, 0), K), ix(I, np.minimum(J + 1, N - 1), K), J - 2 >= 0, J + 1 <= N - 1, hy)
    if Nz > 1:
        I, J, K = mg(np.arange(N), np.arange(N), np.arange(1, Nz))
        faceset(uz[I, J, K - 1], ix(I, J, K - 1), ix(I, J, K), ix(I, J, np.maximum(K - 2, 0)), ix(I, J, np.minimum(K + 1, Nz - 1)), K - 2 >= 0, K + 1 <= Nz - 1, hz)
    M = N * N * Nz
    return sps.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(M, M)).tocsr()


class ScalarNet3:
    """phi(x) = sum_k c_k tanh(w_k.x + b_k) at the cell centres; state beta with phi_c = G beta (c = Phi^T beta; Phi is never stored).
    Kernel-form solve of  B phi_c = rhs  (B = a0/dt I + A)."""

    def __init__(self, N, Nz, cfg):
        self.N, self.Nz, self.cfg = N, Nz, cfg; M = N * N * Nz; self.M = M; self.P = int(cfg["ratio"] * M)
        g = (np.arange(N) + 0.5) / N; gz = (np.arange(Nz) + 0.5) * ZR / Nz
        self.pts = torch.as_tensor(np.stack(np.meshgrid(g, g, gz, indexing="ij"), -1).reshape(-1, 3), device=dev, dtype=DT)
        self.W, self.b = make_features3(self.P, cfg["s_min"], cfg["s_max_cells"] * N, cfg["seed"] + 7)
        t = time.time(); self.G = kernel_blocks([self.pts], self.W, self.b)[0]
        self.fact = None; self.n_fact = 0
        print(f"scalar network: {self.P:,} tanh features, {M:,} cell-centre equations, kernel built in {time.time() - t:.1f} s", flush=True)

    def fit(self, phi_c):
        """beta reproducing given cell-centre values (min-norm): beta = (G + lam I)^-1 phi_c."""
        G = self.G.clone(); G.diagonal().add_(self.cfg["lam_scalar"] * self.G.diagonal().mean())
        L, info = torch.linalg.cholesky_ex(G); out = chol_solve(L, torch.as_tensor(phi_c, device=dev, dtype=DT)); del G, L
        return out

    def values(self, beta):
        return (self.G @ beta).cpu().numpy()

    def weights(self, beta):
        """output weights c = Phi^T beta of the network (computed on demand)."""
        c = torch.zeros(self.P, device=dev, dtype=DT)
        for p0 in range(0, self.P, 4096):
            c[p0:p0 + 4096] = torch.tanh(self.pts @ self.W[:, p0:p0 + 4096] + self.b[p0:p0 + 4096]).T @ beta
        return c

    def _factor(self, Bm):
        """K = B G B^T + lam scale I, Cholesky in place (K -> L)."""
        self.fact = None
        if dev.type == "cuda": torch.cuda.empty_cache()
        Bt = _t(Bm); BtT = _t(Bm.T.tocsr()); M = self.M; ch = 1024; lam = self.cfg["lam_scalar"]; scale = None
        for attempt in range(14):
            K = sym_colmajor(M, DT)
            for r0 in range(0, M, ch):
                Bb = torch.sparse.mm(_t(Bm[r0:r0 + ch]), self.G)
                K[:, r0:r0 + ch] = torch.sparse.mm(Bt, Bb.T.contiguous()); del Bb
            if scale is None: scale = K.diagonal().mean().item()
            K.diagonal().add_(lam * scale)
            if chol_inplace(K) == 0: break
            del K; lam *= 10
        self.fact = (BtT, K); self.n_fact += 1

    def solve(self, Bm, rhs, b0=None):
        """beta with  B (G beta) = rhs, by residual-corrected kernel solves; the factor is re-used while it contracts quickly."""
        Bt = _t(Bm); rhs_t = torch.as_tensor(rhs, device=dev, dtype=DT)
        beta = torch.zeros(self.M, device=dev, dtype=DT) if b0 is None else b0.clone()
        r = rhs_t - _spmv(Bt, self.G @ beta); rn = r.abs().max().item()
        tol = 1e-10 * max(1.0, rhs_t.abs().max().item()); fresh = False
        for it in range(60):
            if rn < tol:
                break
            if self.fact is None:
                self._factor(Bm); fresh = True
            BtT, L = self.fact
            y = chol_solve(L, r)
            bn = beta + _spmv(BtT, y)
            rnew = rhs_t - _spmv(Bt, self.G @ bn); rnn = rnew.abs().max().item()
            if rnn < rn:
                slow = rnn > 0.3 * rn
                beta, r, rn = bn, rnew, rnn
                if slow and not fresh and it >= 3:
                    self.fact = None
            else:
                if fresh:
                    break
                self.fact = None
            if fresh and rn >= tol and it >= 3:
                fresh = False
        return beta, rn


# =============================================================== scalar run on the stored flow history
def circle_mask3(N, Nz):
    g = (np.arange(N) + 0.5) / N * L_; gz = (np.arange(Nz) + 0.5) * DEPTH / Nz
    X, Y, Z = np.meshgrid(g, g, gz, indexing="ij")
    return np.hypot(X - PHYS["cx"], Y - PHYS["cy"]) <= PHYS["r"], X, Y, Z                        # circle extruded through the whole depth


def _direct_solve(Bm, rhs, x0, lu):
    from scipy.sparse.linalg import splu
    tol = 1e-10 * max(1.0, float(np.abs(rhs).max())); x = x0.copy(); used = 0; Bc = Bm.tocsc()
    for it in range(12):
        r = rhs - Bm @ x; rn = float(np.abs(r).max())
        if rn < tol:
            break
        if lu is None:
            lu = splu(Bc)
        x = x + lu.solve(r); used += 1
        if used > 3:
            lu = None
    else:
        r = rhs - Bm @ x; rn = float(np.abs(r).max())
    return x, rn, (lu if used <= 3 else None)


def faces_at(flow, t):
    """Face velocities at time t (linear interpolation between the stored levels; frozen after the last one)."""
    O = flow["O"]; tp = flow["t_psi"]
    if t >= tp[-1] - 1e-12: return O[-1]
    i = int(np.searchsorted(tp, t, side="right") - 1); i = max(i, 0); f = (t - tp[i]) / (tp[i + 1] - tp[i])
    return O[i] if f < 1e-12 else (1 - f) * O[i] + f * O[i + 1]


def run_scalar(flow, N, Nz, cfg, t_inj, t_mix, label, D=None, solver=None, n_frames=None, dt_s=None):
    dt_s = dt_s or cfg["dt"] / cfg["scalar_sub"]; dts = dt_s / T_CONV
    D = (cfg.get("D") or PHYS["D"]) if D is None else D; Dstar = D / (U_ * L_); solver = solver or cfg["scalar_solver"]
    nsteps = int(round(t_mix / dt_s)); n_frames = cfg["n_frames"] if n_frames is None else n_frames
    mask, X, Y, Z = circle_mask3(N, Nz); sn = ScalarNet3(N, Nz, cfg) if solver == "network" else None
    M = N * N * Nz; cvol = (L_ / N) ** 2 * (DEPTH / Nz)
    phi = mask.astype(float).ravel(); phim1 = phi.copy(); beta = sn.fit(phi) if sn else None
    print(f"[{label}] inject at t = {t_inj:.3f} s: {int(mask.sum())} cells with phi = 1 (volume {mask.sum() * cvol * 1e9:.2f} mm^3, exact cylinder "
          f"{np.pi * PHYS['r'] ** 2 * DEPTH * 1e9:.2f} mm^3); D = {D:.2e} m2/s (Pe = {1 / Dstar:.0f}), scalar dt = {dt_s} s, solver = {solver}", flush=True)
    rec = dict(t=[], mi=[], mi_raw=[], mean=[], dmean=[], cx=[], cy=[], cz=[], pmax=[], pmin=[], mass=[], res=[])
    frames = []; fr_steps = set(np.unique(np.linspace(0, nsteps, n_frames).round().astype(int))) if n_frames > 0 else set()
    A_frozen = None; dmean0 = None; t0 = time.time(); lu = None; Xr, Yr, Zr = X.ravel(), Y.ravel(), Z.ravel()

    def monitor(k, m, res):
        nonlocal dmean0
        mean = m.mean(); dmean = np.abs(m - mean).mean(); w = m.sum()
        if dmean0 is None: dmean0 = dmean
        mi_raw = 1.0 - dmean / dmean0
        rec["t"].append(t_inj + k * dt_s); rec["mean"].append(mean); rec["dmean"].append(dmean); rec["mi_raw"].append(mi_raw); rec["mi"].append(min(max(mi_raw, 0.0), 1.0))
        rec["cx"].append((m * Xr).sum() / w); rec["cy"].append((m * Yr).sum() / w); rec["cz"].append((m * Zr).sum() / w)
        rec["pmax"].append(m.max()); rec["pmin"].append(m.min()); rec["mass"].append(m.sum() * cvol); rec["res"].append(res)
        if k in fr_steps:
            frames.append((t_inj + k * dt_s, m.reshape(N, N, Nz).copy(), faces_at(flow, t_inj + k * dt_s).copy()))
    monitor(0, phi, 0.0)
    for k in range(1, nsteps + 1):
        frozen = (t_inj + k * dt_s) >= flow["t_psi"][-1] - 1e-12                          # flow no longer changes: the transport operator is constant
        if not (frozen and A_frozen is not None):
            ux, uy, uz = [a.numpy() for a in split_faces(torch.as_tensor(faces_at(flow, t_inj + k * dt_s)), N, Nz)]
            A = scalar_operator3(ux, uy, uz, N, Nz, Dstar, cfg["adv"]); A_frozen = A if frozen else None
        else:
            A = A_frozen
        a0, hist = (1.0, -phi) if k == 1 else (1.5, -2.0 * phi + 0.5 * phim1)
        Bm = (A + sps.identity(M, format="csr") * (a0 / dts)).tocsr(); rhs = -hist / dts
        if solver == "network":
            beta, res = sn.solve(Bm, rhs, b0=beta); phin = sn.values(beta)
        else:
            phin, res, lu = _direct_solve(Bm, rhs, 2.0 * phi - phim1 if k > 1 else phi, lu)
        phim1, phi = phi, phin
        monitor(k, phi, res)
        if k % max(1, nsteps // 8) == 0:
            print(f"   [{label}] t = {t_inj + k * dt_s:6.2f} s  MI {rec['mi'][-1]:.3f}  phi_max {rec['pmax'][-1]:.3f}  mass drift {rec['mass'][-1] / rec['mass'][0] - 1:+.1e}  "
                  f"linear-solve residual {res:.1e}  ({time.time() - t0:.0f} s)", flush=True)
    for k_ in list(rec): rec[k_] = np.array(rec[k_])
    rec["frames"] = frames; rec["sn"] = sn; rec["dmean0"] = dmean0; rec["D"] = D; rec["dt_s"] = dt_s; rec["n_steps"] = nsteps; rec["t_inj"] = t_inj
    rec["stats"] = dict(features=int(sn.P) if sn else 0, equations=M, factorisations=int(sn.n_fact) if sn else 0, steps=nsteps)
    if sn is not None: sn.fact = None
    return rec
def write_csv(path, header, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for row in zip(*cols):
            w.writerow([f"{x:.9e}" if isinstance(x, (float, np.floating)) else x for x in row])


def read_numeric_table(path):
    """Robust reader for CSV / whitespace / Fluent-XY / CFD-Post exports -> (column names or None, float matrix)."""
    names, rows = None, []
    with open(path, errors="ignore") as f:
        for ln in f:
            ln = ln.strip().lstrip("\ufeff")
            if not ln:
                continue
            toks = [p for p in re.split(r"[,\s;()]+", ln) if p]
            try:
                rows.append([float(x) for x in toks])
            except ValueError:
                mm = re.match(r'^\(labels\s+(.*)\)\s*$', ln)                       # Fluent XY:  (labels "x" "y")
                if mm:
                    names = re.findall(r'"([^"]*)"', mm.group(1))
                elif not rows and names is None and ("," in ln or "\t" in ln):         # csv header
                    names = [p.strip().strip('"') for p in re.split(r"[,\t]", ln)]
    if not rows:
        raise ValueError(f"no numeric rows in {path}")
    ncol = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    data = np.array([r for r in rows if len(r) == ncol], dtype=float)
    if names is not None and len(names) != ncol:
        names = None
    return names, data


def load_csv(path):
    """Structured array with the header names (c0, c1, ... if there is no header)."""
    try:
        names, data = read_numeric_table(path)
        nm = [re.sub(r"\W+", "_", n).strip("_") or f"c{i}" for i, n in enumerate(names)] if names else [f"c{i}" for i in range(data.shape[1])]
        d = np.zeros(len(data), dtype=[(n, float) for n in nm])
        for i, n in enumerate(nm):
            d[n] = data[:, i]
        return d
    except Exception as e:
        print("could not read", path, e); return None


# =============================================================== outputs, validation, training figure, cost report
from scipy.interpolate import PchipInterpolator
REPORT = []


def say(s=""):
    print(s, flush=True); REPORT.append(s)


def show():
    if not matplotlib.get_backend().lower().startswith("agg"): plt.show()


# Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411, Re = 1000: u along x = 0.5 and v along y = 0.5 (Tables I, II); primary vortex psi_min
GHIA_Y = np.array([0, .0547, .0625, .0703, .1016, .1719, .2813, .4531, .5, .6172, .7344, .8516, .9531, .9609, .9688, .9766, 1])
GHIA_X = np.array([0, .0625, .0703125, .078125, .09375, .15625, .2265625, .234375, .5, .8046875, .859375, .90625, .9453125, .953125, .9609375, .96875, 1])
GHIA_U1000 = np.array([0, -.18109, -.20196, -.22220, -.29730, -.38289, -.27805, -.10648, -.06080, .05702, .18719, .33304, .46604, .51117, .57492, .65928, 1])
GHIA_V1000 = np.array([0, .27485, .29012, .30353, .32627, .37095, .33075, .32235, .02526, -.31966, -.42665, -.51550, -.39188, -.33714, -.27669, -.21388, 0])
GHIA_PSI_MIN = -0.117929


def rel_l2(a, b):
    return 100 * np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(b), 1e-30)


def ans_table(key):
    fn = ANSYS_FILES.get(key)
    if not fn: return None
    p = fn if os.path.isabs(fn) else os.path.join(RCFD_DIR, fn)
    return load_csv(p) if os.path.exists(p) else None


def ans_col(d, *pats, default=1):
    nm = d.dtype.names
    for p in pats:
        for n in nm:
            if re.search(p, n, re.I): return d[n]
    return d[nm[default]]


def ans_time(d, t_inj, shift):
    t = d[d.dtype.names[0]].astype(float)
    return t + t_inj if (shift and t[0] < 0.5) else t


def mid_layers(Nz):
    return [Nz // 2] if Nz % 2 else [Nz // 2 - 1, Nz // 2]


def mid_profiles(o, N, Nz):
    """u(y) on x = L/2 and v(x) on y = L/2 at the mid-depth of the slab."""
    ux, uy, uz = [a.numpy() for a in split_faces(torch.as_tensor(o), N, Nz)]; kk = mid_layers(Nz)
    um = ux[:, :, kk].mean(2); vm = uy[:, :, kk].mean(2); k = N // 2; c = (np.arange(N) + .5) / N
    return dict(u_y=PchipInterpolator(np.r_[0, c, 1], np.r_[0, um[k - 1, :], 1]), v_x=PchipInterpolator(np.r_[0, c, 1], np.r_[0, vm[:, k - 1], 0]))


def psi_depth_mean(o, N, Nz):
    """In-plane stream function of the depth-averaged flow, on the corner grid (N+1, N+1) [i (x), j (y)]."""
    ux = split_faces(torch.as_tensor(o), N, Nz)[0].numpy().mean(2); psi = np.zeros((N + 1, N + 1)); psi[1:N, 1:] = np.cumsum(ux / N, axis=1)
    return psi


def ghia_errors(o, N, Nz):
    f = mid_profiles(o, N, Nz); inn = slice(1, -1)
    return rel_l2(f["u_y"](GHIA_Y[inn]), GHIA_U1000[inn]), rel_l2(f["v_x"](GHIA_X[inn]), GHIA_V1000[inn]), abs(psi_depth_mean(o, N, Nz).min() - GHIA_PSI_MIN) / abs(GHIA_PSI_MIN) * 100


def pretty(x): return f"{x:g}"


# ---------------------------------------------------------------- figures
def fig_series(flow, rec, N, Nz, Re, label, folder):
    ke_a, mi_a, cog_a = ans_table("ke"), ans_table("mi"), ans_table("cog"); t_inj = rec["t_inj"]
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 8.4))
    ax[0, 0].plot(flow["t"], flow["KE"] * 1e9, color=C_SPIRAL, label=NAME)
    if ke_a is not None: ax[0, 0].plot(ans_time(ke_a, 0, False), ans_col(ke_a, "kinetic|ke") * 1e9, color=C_REF, lw=2.2, zorder=1, label=REF)
    if flow.get("plateau_found", True): ax[0, 0].axvline(flow["t_plateau"], color="tab:green", ls=":", label=(f"KE flat: $t$ = {flow['t_plateau']:.2f} s" if flow.get("plateau_found", True) else "KE not yet flat")); ax[0, 0].axvline(t_inj, color=C_SPIRAL2, ls="--", lw=1, label="injection")
    ax[0, 0].set(xlabel=r"$t$ [s]", ylabel="kinetic energy [nJ]", title="(a) Kinetic energy"); ax[0, 0].legend()
    ax[0, 1].plot(rec["t"] - t_inj, rec["mi"], color=C_SPIRAL, label=NAME)
    if mi_a is not None:
        ta = mi_a[mi_a.dtype.names[0]].astype(float); ax[0, 1].plot(ta - ta[0], np.clip(ans_col(mi_a, "mix"), 0, 1), color=C_REF, lw=2.2, zorder=1, label=REF + " (injected at 4 s)")
    ax[0, 1].set(xlabel="time after injection [s]", ylabel="mixing index [$-$]", title="(b) Mixing index", ylim=(-0.02, 1.0)); ax[0, 1].legend()
    ax[1, 0].plot(rec["cx"] * 1e3, rec["cy"] * 1e3, color=C_SPIRAL, label=NAME); ax[1, 0].plot(rec["cx"][0] * 1e3, rec["cy"][0] * 1e3, "o", color="lime", mec="k", label="start")
    ax[1, 0].plot(rec["cx"][-1] * 1e3, rec["cy"][-1] * 1e3, "o", color=C_SPIRAL2, mec="k", label="end")
    if cog_a is not None:
        cx_, cy_ = ans_col(cog_a, "cog_?x|^x", default=1), ans_col(cog_a, "cog_?y|^y", default=2); sc = 1e3 if np.nanmax(np.abs(cx_)) < 1 else 1.0
        ax[1, 0].plot(cx_ * sc, cy_ * sc, color=C_REF, lw=2.2, zorder=1, label=REF)
    ax[1, 0].set(xlabel=r"$x$ [mm]", ylabel=r"$y$ [mm]", title="(c) Centre of gravity of $\\phi$"); ax[1, 0].set_aspect("equal"); ax[1, 0].legend()
    ax[1, 1].plot(rec["t"] - t_inj, rec["pmax"], color=C_SPIRAL, label=r"$\phi_{\max}$"); ax[1, 1].set(xlabel="time after injection [s]", ylabel=r"$\phi_{\max}$ [$-$]", title=r"(d) Peak of $\phi$ and conservation")
    b = ax[1, 1].twinx(); b.plot(rec["t"] - t_inj, rec["mass"] / rec["mass"][0] - 1, color=C_SPIRAL2, label=r"$\int\phi\,dV$ drift"); b.set_ylabel("relative drift of total $\\phi$")
    h1, l1 = ax[1, 1].get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); ax[1, 1].legend(h1 + h2, l1 + l2, loc="center right")
    fig.suptitle(f"{NAME}-3D  |  {label}  |  Re = {Re:.0f}  |  {N}$\\times${N}$\\times${Nz} cells, front/back: {Z_WALL}", fontsize=11)
    fig.tight_layout(); save_fig(fig, f"{folder}/Fig_KE_MI_CoG"); show(); plt.close(fig)


def fig_centrelines(flow, N, Nz, Re, folder):
    """u(y) on x = L/2 and v(x) on y = L/2 at mid-depth, final time; Ghia et al. (1982) for Re = 1000 (2D cavity) and, if present, the ANSYS Reference."""
    o = flow["O"][-1]; f = mid_profiles(o, N, Nz); xs = np.linspace(0, 1, 801); ua, va = ans_table("uvel"), ans_table("vvel")
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.9))
    ax[0].plot(f["u_y"](xs), xs, color=C_SPIRAL, ls="--", lw=2.0, label=NAME, zorder=3); ax[1].plot(xs, f["v_x"](xs), color=C_SPIRAL, ls="--", lw=2.0, label=NAME, zorder=3)
    if abs(Re - 1000) < 1:
        ax[0].plot(GHIA_U1000, GHIA_Y, "s", mfc="none", mec="k", ms=6, label=GHIA_LAB, zorder=4); ax[1].plot(GHIA_X, GHIA_V1000, "s", mfc="none", mec="k", ms=6, label=GHIA_LAB, zorder=4)
    out = {}

    def norm_coord(x): x = np.asarray(x, float); return x / L_ if x.max() <= 0.0501 else (x if x.max() <= 1.001 else x / (L_ * 1e3))
    for d, a, which in ((ua, ax[0], "u"), (va, ax[1], "v")):
        if d is None: continue
        nm = d.dtype.names; x = norm_coord(d[nm[0]]); v = d[nm[1]].astype(float); v = v / U_ if np.abs(v).max() > 1.5 else v; o_ = np.argsort(x)
        if which == "u": a.plot(v[o_], x[o_], color=C_REF, lw=2.8, label=REF, zorder=1); out["u"] = (x[o_], v[o_])
        else: a.plot(x[o_], v[o_], color=C_REF, lw=2.8, label=REF, zorder=1); out["v"] = (x[o_], v[o_])
    inn = slice(1, -1)
    if abs(Re - 1000) < 1:
        say(f"\nmid-line validation at t = {flow['t'][-1]:.2f} s, mid-depth, {N}x{N}x{Nz}, front/back {Z_WALL}:  {NAME} vs {GHIA_LAB} (2D cavity):  "
            f"u(y): rel. L2 {rel_l2(f['u_y'](GHIA_Y[inn]), GHIA_U1000[inn]):.2f} %,  v(x): rel. L2 {rel_l2(f['v_x'](GHIA_X[inn]), GHIA_V1000[inn]):.2f} %")
        if Z_WALL == "noslip": say("   (Ghia et al. is a 2D cavity: with no-slip front/back walls 5 mm apart the 3D flow is expected to differ - this is not an error of the solver)")
    for key, fs, nm_ in (("u", f["u_y"], "u(y)"), ("v", f["v_x"], "v(x)")):
        if key in out:
            xr, vr = out[key]; ii = (xr > 1e-6) & (xr < 1 - 1e-6)
            say(f"   {NAME} vs {REF}: {nm_}: rel. L2 {rel_l2(fs(xr[ii]), vr[ii]):.2f} %, max |difference| {np.abs(fs(xr[ii]) - vr[ii]).max():.4f} U")
    ax[0].set(xlabel=r"$u/U$ [$-$]", ylabel=r"$y/L$", title=r"(a) $u$ on the vertical mid-line $x=L/2$"); ax[1].set(xlabel=r"$x/L$", ylabel=r"$v/U$ [$-$]", title=r"(b) $v$ on the horizontal mid-line $y=L/2$")
    for a in ax: a.legend(fontsize=8)
    fig.suptitle(f"Mid-depth mid-line velocity profiles, Re = {Re:.0f}, {N}$\\times${N}$\\times${Nz}, front/back {Z_WALL}, $t$ = {flow['t'][-1]:.1f} s", fontsize=11)
    fig.tight_layout(); save_fig(fig, f"{folder}/Fig_centrelines"); show(); plt.close(fig)


def fig_slices(flow, N, Nz, Re, folder):
    """Where the 3D structure is: speed on the mid-depth plane, u on the y-z section x = L/2, and the profile of u across the depth."""
    o = torch.as_tensor(flow["O"][-1]); ux, uy, uz = [a.numpy() for a in split_faces(o, N, Nz)]; uc, vc, wc = [a.numpy() for a in cell_velocity(o, N, Nz)]
    kk = mid_layers(Nz); spd = np.sqrt(uc ** 2 + vc ** 2 + wc ** 2); psi = psi_depth_mean(flow["O"][-1], N, Nz); gx = np.linspace(0, L_ * 1e3, N + 1)
    fig = plt.figure(figsize=(15.5, 4.8)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 0.85], wspace=0.32)
    a = fig.add_subplot(gs[0]); im = a.imshow(spd[:, :, kk].mean(2).T, origin="lower", extent=(0, L_ * 1e3, 0, L_ * 1e3), cmap="viridis"); a.contour(gx, gx, psi.T, levels=np.linspace(psi.min(), -1e-4, 9), colors="w", linewidths=0.5, alpha=0.7)
    fig.colorbar(im, ax=a, fraction=0.046, label=r"$|\mathbf{u}|/U$"); a.set(xlabel=r"$x$ [mm]", ylabel=r"$y$ [mm]", title="(a) Speed, mid-depth plane"); a.grid(False)
    b = fig.add_subplot(gs[1]); k = N // 2; zc = (np.arange(Nz) + .5) * DEPTH / Nz * 1e3
    im = b.imshow(uc[k, :, :].T, origin="lower", extent=(0, L_ * 1e3, 0, DEPTH * 1e3), cmap="RdBu_r", aspect="auto", vmin=-np.abs(uc[k]).max(), vmax=np.abs(uc[k]).max())
    fig.colorbar(im, ax=b, fraction=0.046, label=r"$u/U$"); b.set(xlabel=r"$y$ [mm]", ylabel=r"$z$ [mm] (depth)", title=r"(b) $u$ on the section $x=L/2$"); b.grid(False)
    c = fig.add_subplot(gs[2])
    for yy, col in ((0.9, C_SPIRAL2), (0.5, C_SPIRAL), (0.1, C_SPIRAL3)):
        j = int(yy * N); c.plot(uc[k, j, :], zc, "o-", color=col, label=rf"$y/L$ = {yy}")
    c.set(xlabel=r"$u/U$ at $x=L/2$", ylabel=r"$z$ [mm]", title="(c) Profile across the depth"); c.legend(fontsize=8)
    c.text(0.03, 0.03, rf"max $|w|/U$ = {np.abs(wc).max():.1e}", transform=c.transAxes, fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    fig.suptitle(f"{NAME}-3D flow structure at $t$ = {flow['t'][-1]:.1f} s, Re = {Re:.0f}, front/back faces: {Z_WALL}", fontsize=11)
    save_fig(fig, f"{folder}/Fig_flow_slices"); show(); plt.close(fig)


GHIA_VORT = dict(primary=(0.5313, 0.5625, -0.117929), bottom_right=(0.8594, 0.1094, 1.75102e-3), bottom_left=(0.0859, 0.0781, 2.31129e-4))      # Ghia et al. (1982), Re = 1000, Table III
PSI_NEG = [-1e-10, -1e-7, -1e-5, -1e-4, -0.01, -0.03, -0.05, -0.07, -0.09, -0.10, -0.11, -0.115, -0.1175]                                        # contour levels of Ghia et al. (1982), Fig. 3
PSI_POS = [1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3, 3e-3]


def vortex_centres(psi, N):
    """Primary vortex (global minimum of psi) and the two bottom-corner vortices (local maxima near the corners), from a cubic spline of the corner values."""
    from scipy.interpolate import RectBivariateSpline
    g = np.linspace(0, 1, N + 1); spl = RectBivariateSpline(g, g, psi, kx=3, ky=3); f = np.linspace(0, 1, 8 * N + 1); P = spl(f, f)
    out = {}
    i, j = np.unravel_index(np.argmin(P), P.shape); out["primary"] = (f[i], f[j], P[i, j])
    for name, (x0, x1) in (("bottom_right", (0.65, 1.0)), ("bottom_left", (0.0, 0.35))):
        ii = (f >= x0) & (f <= x1); jj = f <= 0.35; sub = P[np.ix_(ii, jj)]; a, b = np.unravel_index(np.argmax(sub), sub.shape)
        out[name] = (f[ii][a], f[jj][b], sub[a, b])
    return out


def fig_vortex_vectors(flow, N, Nz, Re, folder):
    """Stream function and velocity vectors on the mid-depth plane: primary vortex and the two bottom-corner vortices."""
    o = flow["O"][-1]; uc, vc, wc = [a.numpy() for a in cell_velocity(torch.as_tensor(o), N, Nz)]; kk = mid_layers(Nz); um = uc[:, :, kk].mean(2); vm = vc[:, :, kk].mean(2)
    psi = psi_depth_mean(o, N, Nz); g = np.linspace(0, 1, N + 1); cc = (np.arange(N) + .5) / N; X, Y = np.meshgrid(cc, cc, indexing="ij"); spd = np.hypot(um, vm)
    vc_ = vortex_centres(psi, N)
    fig, ax = plt.subplots(2, 2, figsize=(13.4, 12.2)); ax = ax.ravel()

    def levels(a, lv, sign):
        lv = [l for l in lv if (l < 0) == (sign < 0) and (psi.min() <= l if sign < 0 else l <= psi.max())]
        if lv: a.contour(g, g, psi.T, levels=sorted(lv), colors="tab:blue" if sign < 0 else "tab:red", linewidths=0.9)
    # (a) whole cavity: stream function + vectors
    a = ax[0]; im = a.pcolormesh(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1), spd.T, cmap="Greys", vmin=0, vmax=max(spd.max(), 1e-9), shading="flat", alpha=0.55)
    levels(a, PSI_NEG, -1); levels(a, PSI_POS, +1); s = max(1, N // 25)
    a.quiver(X[::s, ::s], Y[::s, ::s], um[::s, ::s], vm[::s, ::s], color="k", pivot="mid", width=0.0022, scale=None)
    for nm, (x, y, p) in vc_.items(): a.plot(x, y, "x" if nm == "primary" else "o", color="tab:green", ms=9, mew=2, mfc="none")
    fig.colorbar(im, ax=a, fraction=0.046, label=r"in-plane speed $/U$ (mid-depth)"); a.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=r"$x/L$", ylabel=r"$y/L$", title=r"(a) Stream function $\psi$ (depth-mean flow) and velocity vectors (mid-depth plane)"); a.grid(False)
    # (b) streamlines of the mid-plane velocity
    a = ax[1]; a.streamplot(cc, cc, um.T, vm.T, color=spd.T, cmap="viridis", density=2.2, linewidth=1.0, arrowsize=0.9); a.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=r"$x/L$", ylabel=r"$y/L$", title="(b) Streamlines of the mid-depth in-plane velocity"); a.grid(False)
    # (c), (d) corner vortices, zoomed
    for a, (nm, lab, box) in zip(ax[2:], (("bottom_right", "bottom-right", (0.65, 1.0, 0.0, 0.35)), ("bottom_left", "bottom-left", (0.0, 0.35, 0.0, 0.35)))):
        ii = (cc >= box[0]) & (cc <= box[1]); jj = (cc >= box[2]) & (cc <= box[3])
        a.pcolormesh(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1), spd.T, cmap="Greys", vmin=0, vmax=max(spd[np.ix_(ii, jj)].max(), 1e-9), shading="flat", alpha=0.45)
        levels(a, PSI_POS, +1); levels(a, PSI_NEG, -1)
        a.quiver(X[np.ix_(ii, jj)], Y[np.ix_(ii, jj)], um[np.ix_(ii, jj)], vm[np.ix_(ii, jj)], color="k", pivot="mid", width=0.004)
        x, y, p = vc_[nm]; a.plot(x, y, "o", color="tab:green", ms=10, mew=2, mfc="none", label=rf"SPIRAL: ({x:.3f}, {y:.3f}), $\psi$ = {p:.2e}")
        gx, gy, gp = GHIA_VORT[nm]; a.plot(gx, gy, "s", color="k", ms=8, mfc="none", label=rf"{GHIA_LAB}: ({gx:.4f}, {gy:.4f}), $\psi$ = {gp:.2e}")
        a.set(xlim=box[:2], ylim=box[2:], aspect="equal", xlabel=r"$x/L$", ylabel=r"$y/L$", title=f"({'c' if nm == 'bottom_right' else 'd'}) {lab} corner vortex (zoom)"); a.legend(fontsize=8, loc="upper center", frameon=True, framealpha=0.9); a.grid(False)
    fig.suptitle(f"{NAME}-3D vortices at $t$ = {flow['t'][-1]:.2f} s, Re = {Re:.0f}, {N}$\\times${N}$\\times${Nz}, front/back {Z_WALL}  (blue: clockwise, $\\psi<0$;  red: counter-clockwise, $\\psi>0$)", fontsize=11)
    fig.tight_layout(); save_fig(fig, f"{folder}/Fig_vortices_vectors_midsection"); show(); plt.close(fig)


def fig_vortices(flow, N, Nz, Re, folder):
    """Primary vortex and the two bottom-corner eddies of the depth-mean stream function (mid-section), in the style of the 2D 'cavity-flow check':
    (a) psi with the black primary-vortex levels and the magenta corner-eddy levels of Ghia et al. (1982), (b) streamlines coloured by |V|/U, (c) bottom corners."""
    from scipy.interpolate import RectBivariateSpline
    o = flow["O"][-1]; psi = psi_depth_mean(o, N, Nz); g = np.linspace(0, 1, N + 1); spl = RectBivariateSpline(g, g, psi, kx=3, ky=3)
    f = np.linspace(0, 1, 8 * N + 1); P = spl(f, f); U_ = spl(f, f, dy=1); V_ = -spl(f, f, dx=1); X, Y = np.meshgrid(f, f, indexing="ij"); spd = np.hypot(U_, V_)
    vc_ = vortex_centres(psi, N); pmin, pmax = P.min(), P.max()
    neg = sorted(l for l in PSI_NEG if pmin <= l < 0); pos = sorted(l for l in PSI_POS if l <= pmax)
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.9))
    for a in ax: a.set(aspect="equal", xlabel=r"$x/L$", ylabel=r"$y/L$"); a.grid(False)
    a = ax[0]; cs = a.contourf(X, Y, P, levels=np.linspace(pmin, 0, 60), cmap="Blues_r"); fig.colorbar(cs, ax=a, label=r"$\psi\,/\,(U\,L)$", fraction=0.046, pad=0.03)
    if neg: a.contour(X, Y, P, levels=neg, colors="k", linewidths=0.8, linestyles="dashed")
    if pos: a.contour(X, Y, P, levels=pos, colors="m", linewidths=0.8)
    x, y, p = vc_["primary"]; gx, gy, gp = GHIA_VORT["primary"]
    a.plot(x, y, "*", color="w", mec="k", ms=15, mew=1.0); a.plot(gx, gy, "kx", ms=9, mew=1.8)
    a.set(xlim=(0, 1), ylim=(0, 1)); a.set_title(rf"stream function, $\psi_{{\min}}$ = {p:.5f}  at ({x:.3f}, {y:.3f})", fontsize=10.5)
    a = ax[1]; st = a.streamplot(f, f, U_.T, V_.T, color=spd.T, cmap="jet", density=2.6, linewidth=0.9, arrowsize=0.8); a.set(xlim=(0, 1), ylim=(0, 1)); a.set_title(r"streamlines (colour = $|\mathbf{V}|/U$)", fontsize=10.5)
    a = ax[2]
    if len(pos) >= 1:
        lv = pos + [max(pmax, pos[-1]) * 1.001]
        a.contourf(X, Y, np.where(P > 0, P, np.nan), levels=lv, cmap="Purples"); a.contour(X, Y, P, levels=pos, colors="m", linewidths=1.0)
    for nm in ("bottom_right", "bottom_left"):
        x, y, p = vc_[nm]; gx, gy, gp = GHIA_VORT[nm]
        if p > 0: a.plot(x, y, "*", color="m", mec="k", ms=13)
        a.plot(gx, gy, "kx", ms=9, mew=1.8)
    a.set(xlim=(0, 1), ylim=(0, 0.45)); a.set_title(r"bottom corners ($\psi>0$ = counter-rotating eddies)", fontsize=10.5)
    a.text(0.03, 0.40, rf"$\star$ {NAME}-3D     $\times$ {GHIA_LAB} (steady)", fontsize=9)
    fig.suptitle(f"Cavity-flow check: {N}$\\times${N}$\\times${Nz} cells, front/back {Z_WALL}, Re = {Re:.0f}, $t$ = {flow['t'][-1]:.1f} s.   Black contours: primary vortex, magenta: corner eddies (Ghia levels)", fontsize=12)
    fig.tight_layout(); save_fig(fig, f"{folder}/Fig_vortices_midsection"); show(); plt.close(fig)
    say(f"\nvortex centres on the mid section at t = {flow['t'][-1]:.2f} s (stream function of the depth-mean flow, cubic-spline extrema), {N}x{N}x{Nz}, front/back {Z_WALL}:")
    say(f"  {'vortex':14s} {'SPIRAL x, y':>16} {'psi':>12} | {GHIA_LAB + ' (2D) x, y':>30} {'psi':>12}")
    for nm in ("primary", "bottom_right", "bottom_left"):
        x, y, p = vc_[nm]; gx, gy, gp = GHIA_VORT[nm]; say(f"  {nm:14s} {x:7.4f},{y:7.4f} {p:12.4e} | {gx:14.4f},{gy:7.4f} {gp:12.4e}")
    if Z_WALL == "noslip": say("  (the reference is a 2D cavity; with no-slip front/back walls 5 mm apart the vortices are weaker and shifted - expected, not a solver error)")
    if abs(Re - 1000) > 1: say("  (Ghia et al. values are for Re = 1000)")


def fig_phi_sequence(rec, flow, N, Nz, Re, label, folder):
    """18 snapshots of phi (mid-depth plane, fixed scale 0-1), like Fig. 5 of the PDF."""
    fr = rec["frames"]
    if not fr: return
    pick = np.unique(np.linspace(0, len(fr) - 1, 18).round().astype(int)); kk = mid_layers(Nz)
    nr = int(np.ceil(len(pick) / 6)); fig, axs = plt.subplots(nr, 6, figsize=(15.5, 2.8 * nr), squeeze=False)
    for a in axs.ravel(): a.axis("off")
    for a, i in zip(axs.ravel(), pick):
        t, ph3, _ = fr[i]; ph = ph3[:, :, kk].mean(2); a.axis("on")
        im = a.imshow(ph.T, origin="lower", extent=(0, L_ * 1e3, 0, L_ * 1e3), cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
        a.set_title(rf"$t$ = {t:.2f} s  ($\phi_{{\max}}$ = {ph.max():.2f})", fontsize=9); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01).set_label(r"$\phi$ [$-$]")
    fig.suptitle(f"{NAME}-3D: evolution of $\\phi$ at mid-depth ({label}), Re = {Re:.0f}, {N}$\\times${N}$\\times${Nz} cells", fontsize=12); save_fig(fig, f"{folder}/phi_sequence", dpi=200); show(); plt.close(fig)


# ---------------------------------------------------------------- 3D views: isometric velocity contours and 3D streamlines
def padded_field(o, N, Nz):
    """Trilinear interpolators of the three velocity components including the wall values (lid u = 1 on top, no-slip walls 0, symmetry planes: zero gradient)."""
    from scipy.interpolate import RegularGridInterpolator as RGI
    ux, uy, uz = [a.numpy() for a in split_faces(torch.as_tensor(o), N, Nz)]; h = 1.0 / N; hz = ZR / Nz; sym = (Z_WALL == "symmetry")
    xc = np.r_[0, (np.arange(N) + .5) * h, 1.0]; xf = np.arange(N + 1) * h; zc = np.r_[0, (np.arange(Nz) + .5) * hz, ZR]; zf = np.arange(Nz + 1) * hz
    a = np.zeros((N + 1, N + 2, Nz + 2)); a[1:N, 1:N + 1, 1:Nz + 1] = ux
    b = np.zeros((N + 2, N + 1, Nz + 2)); b[1:N + 1, 1:N, 1:Nz + 1] = uy
    c = np.zeros((N + 2, N + 2, Nz + 1)); c[1:N + 1, 1:N + 1, 1:Nz] = uz
    if sym:
        for t in (a, b): t[:, :, 0] = t[:, :, 1]; t[:, :, -1] = t[:, :, -2]
    a[1:N, N + 1, :] = 1.0                                                               # the lid moves at every depth
    fu, fv, fw = RGI((xf, xc, zc), a), RGI((xc, xf, zc), b), RGI((xc, xc, zf), c)

    def vel(P):
        P = np.asarray(P, float).copy(); P[:, 0] = np.clip(P[:, 0], 0, 1); P[:, 1] = np.clip(P[:, 1], 0, 1); P[:, 2] = np.clip(P[:, 2], 0, ZR)
        return np.stack([fu(P), fv(P), fw(P)], 1)
    return vel


def three_d_report(o, N, Nz, Re):
    uc, vc, wc = [a.numpy() for a in cell_velocity(torch.as_tensor(o), N, Nz)]
    zvar = np.abs(uc.max(2) - uc.min(2)).max(); wmax = np.abs(wc).max()
    say(f"\nhow 3D is the flow?  max |w|/U = {wmax:.2e},  max variation of u across the depth = {zvar:.2e} U  (front/back faces '{Z_WALL}')" +
        ("  -> exactly depth-independent by construction (symmetry planes): the 3D result equals the 2D one." if Z_WALL == "symmetry" else "  -> genuinely three-dimensional (wall friction of the front/back plates)."))
    return wmax, zvar


def _box_lines(ax, Lx, D, Ly):
    for (x0, x1) in ((0, Lx),):
        for yy in (0, D):
            for zz in (0, Ly):
                ax.plot([x0, x1], [yy, yy], [zz, zz], color="0.25", lw=0.8)
    for xx in (0, Lx):
        for zz in (0, Ly): ax.plot([xx, xx], [0, D], [zz, zz], color="0.25", lw=0.8)
        for yy in (0, D): ax.plot([xx, xx], [yy, yy], [0, Ly], color="0.25", lw=0.8)


def _style3d(ax, Lx, D, Ly, S, title, true_aspect):
    ax.set_xlim(0, Lx); ax.set_ylim(0, D * S); ax.set_zlim(0, Ly); ax.set_box_aspect((Lx, max(D * S, 1e-9), Ly))
    ax.view_init(elev=35.264, azim=-45)                                                  # true isometric view
    ax.set_xlabel(r"$x$ [mm]", labelpad=6); ax.set_zlabel(r"$y$ [mm] (lid on top)", labelpad=4)
    ax.set_ylabel(r"depth $z$ [mm]" + ("" if true_aspect else f" (stretched x{S:g})"), labelpad=8)
    ax.set_yticks([0, D * S / 2, D * S]); ax.set_yticklabels([f"{0:g}", f"{D / 2:g}", f"{D:g}"])
    ax.set_title(title, fontsize=10.5, pad=2); ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis): a.pane.set_facecolor((1, 1, 1, 0.0))


def fig_3d_velocity(flow, N, Nz, Re, folder):
    """Isometric 3D views of the speed |u|/U (banded contours)."""
    from matplotlib import cm, colors
    o = flow["O"][-1]; vel = padded_field(o, N, Nz); Lx = L_ * 1e3; D = DEPTH * 1e3; wmax, zvar = three_d_report(o, N, Nz, Re)
    sp = lambda P: np.linalg.norm(vel(P), axis=1); vmax = float(np.ceil(max(sp(np.stack(np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60), [ZR / 2], indexing="ij"), -1).reshape(-1, 3)).max(), 1e-6) * 20) / 20)
    norm = colors.BoundaryNorm(np.linspace(0, vmax, 16), 256); cmap = plt.get_cmap("turbo"); fcol = lambda v: cmap(norm(v))
    n = 70; xs = np.linspace(0, 1, n); ys = np.linspace(0, 1, n); zs = np.linspace(0, ZR, max(8, 4 * Nz))
    fig = plt.figure(figsize=(16.5, 7.6)); S = max(1.0, round(0.5 * Lx / D / 2)) if False else 6.0
    # (a) stacked layers: x-y planes at every cell layer, depth stretched
    ax = fig.add_subplot(1, 2, 1, projection="3d"); Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    for k in range(Nz):
        zk = (k + 0.5) * ZR / Nz; P = np.stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, zk)], 1); V = sp(P).reshape(Xg.shape)
        ax.plot_surface(Xg * Lx, np.full_like(Xg, zk * L_ * 1e3 * S), Yg * Lx, facecolors=fcol(V), rstride=1, cstride=1, shade=False, antialiased=False, alpha=0.96)
    _box_lines(ax, Lx, D * S, Lx); _style3d(ax, Lx, D, Lx, S, f"(a) speed on the {Nz} cell layers through the depth (depth stretched x{S:g})", False)
    # (b) true proportions: speed on the three visible outer faces of the 50 x 50 x 5 mm box (a hair inside the walls)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d"); e = 0.5 * min(1.0 / N, ZR / Nz)
    Xa, Za = np.meshgrid(xs, zs, indexing="ij")                                           # front face: depth = e  (x-y plane)
    Xf, Yf = np.meshgrid(xs, ys, indexing="ij"); Pf = np.stack([Xf.ravel(), Yf.ravel(), np.full(Xf.size, e)], 1)
    ax2.plot_surface(Xf * Lx, np.full_like(Xf, e * Lx), Yf * Lx, facecolors=fcol(sp(Pf).reshape(Xf.shape)), rstride=1, cstride=1, shade=False, antialiased=False)
    Ys, Zs = np.meshgrid(ys, zs, indexing="ij"); Pr = np.stack([np.full(Ys.size, 1 - e), Ys.ravel(), Zs.ravel()], 1)             # right face: x = L - e
    ax2.plot_surface(np.full_like(Ys, (1 - e) * Lx), Zs * Lx, Ys * Lx, facecolors=fcol(sp(Pr).reshape(Ys.shape)), rstride=1, cstride=1, shade=False, antialiased=False)
    Xt, Zt = np.meshgrid(xs, zs, indexing="ij"); Pt = np.stack([Xt.ravel(), np.full(Xt.size, 1 - e), Zt.ravel()], 1)                # top face: y = L - e  (below the lid)
    ax2.plot_surface(Xt * Lx, Zt * Lx, np.full_like(Xt, (1 - e) * Lx), facecolors=fcol(sp(Pt).reshape(Xt.shape)), rstride=1, cstride=1, shade=False, antialiased=False)
    _box_lines(ax2, Lx, D, Lx); _style3d(ax2, Lx, D, Lx, 1.0, "(b) true proportions 50 x 50 x 5 mm: speed just inside the front, right and top (lid) walls", True)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([]); cb = fig.colorbar(sm, ax=[ax, ax2], fraction=0.025, pad=0.02, shrink=0.8); cb.set_label(r"$|\mathbf{u}|/U$")
    fig.suptitle(f"{NAME}-3D velocity magnitude, isometric view, $t$ = {flow['t'][-1]:.2f} s, Re = {Re:.0f}, {N}$\\times${N}$\\times${Nz} cells, front/back faces: {Z_WALL}   "
                 f"[max $|w|/U$ = {wmax:.1e}, depth variation of $u$ = {zvar:.1e} $U$]", fontsize=11)
    save_fig(fig, f"{folder}/Fig_3D_velocity_isometric", dpi=220); show(); plt.close(fig)


def trace_streamlines(vel, seeds, ds, nmax):
    P = seeds.copy(); alive = np.ones(len(P), bool); paths = [P.copy()]
    f = lambda Q: (lambda V: V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12))(vel(Q))
    for _ in range(nmax):
        k1 = f(P); k2 = f(P + 0.5 * ds * k1); k3 = f(P + 0.5 * ds * k2); k4 = f(P + ds * k3); Pn = P + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        sp_ = np.linalg.norm(vel(Pn), axis=1); inside = (Pn[:, 0] > 0.004) & (Pn[:, 0] < 0.996) & (Pn[:, 1] > 0.004) & (Pn[:, 1] < 0.996) & (Pn[:, 2] > 0) & (Pn[:, 2] < ZR)
        alive &= inside & (sp_ > 2e-3); P = np.where(alive[:, None], Pn, P); pth = P.copy(); pth[~alive] = np.nan; paths.append(pth)
    return np.array(paths).transpose(1, 0, 2)                                            # (seed, step, 3)


def fig_3d_streamlines(flow, N, Nz, Re, folder):
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    o = flow["O"][-1]; vel = padded_field(o, N, Nz); Lx = L_ * 1e3; D = DEPTH * 1e3; S = 6.0
    gx = np.array([0.12, 0.25, 0.4, 0.55, 0.7, 0.85]); gy = np.array([0.12, 0.3, 0.5, 0.7, 0.88]); gz = np.array([0.2, 0.5, 0.8]) * ZR
    seeds = np.array([[x, y, z] for x in gx for y in gy for z in gz]); ds = 0.5 / N; paths = trace_streamlines(vel, seeds, ds, int(3.2 / ds))
    fig = plt.figure(figsize=(16.5, 7.6)); out = []
    for i, (by, lab, cmap_, vlim) in enumerate((("speed", r"$|\mathbf{u}|/U$", "turbo", None), ("depth", r"depth $z$ [mm]", "coolwarm", (0, D)))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d"); segs, cols = [], []
        for p in paths:
            ok = ~np.isnan(p[:, 0]); q = p[ok]
            if len(q) < 3: continue
            spd = np.linalg.norm(vel(q), axis=1); xyz = np.stack([q[:, 0] * Lx, q[:, 2] * L_ * 1e3 * S, q[:, 1] * Lx], 1)
            segs.append(np.stack([xyz[:-1], xyz[1:]], 1)); cols.append(0.5 * (spd[:-1] + spd[1:]) if by == "speed" else 0.5 * (q[:-1, 2] + q[1:, 2]) * L_ * 1e3)
        allc = np.concatenate(cols); nrm = colors.Normalize(*(vlim or (0, allc.max()))); lc = Line3DCollection(np.concatenate(segs), cmap=cmap_, norm=nrm, linewidths=1.1); lc.set_array(allc); ax.add_collection3d(lc)
        _box_lines(ax, Lx, D * S, Lx); _style3d(ax, Lx, D, Lx, S, "(a) 3D streamlines coloured by speed" if i == 0 else "(b) the same streamlines coloured by depth: a straight colour means no motion across the depth", False)
        cb = fig.colorbar(cm.ScalarMappable(norm=nrm, cmap=cmap_), ax=ax, fraction=0.03, pad=0.04, shrink=0.75); cb.set_label(lab)
    uc, vc, wc = [a.numpy() for a in cell_velocity(torch.as_tensor(o), N, Nz)]
    fig.suptitle(f"{NAME}-3D streamlines, isometric view (depth stretched x{S:g}), $t$ = {flow['t'][-1]:.2f} s, Re = {Re:.0f}, front/back faces: {Z_WALL}, max $|w|/U$ = {np.abs(wc).max():.1e}", fontsize=11)
    save_fig(fig, f"{folder}/Fig_3D_streamlines", dpi=220); show(); plt.close(fig)


def training_figure(flow, rec, N, Nz, Re, folder):
    """'Training' information of SPIRAL-3D.  There is no epoch loop: the output weights of the frozen network are found by Newton / Levenberg-Marquardt
    iterations at every time step: (a) loss L = mean(r^2) of the discrete curl-momentum equation versus iteration, (b) loss before / after each time step,
    (c) iterations and factorisations, (d) stopping criterion, (e) error against Ghia et al. (1982) as the flow develops, (f) scalar-network residual and conservation."""
    tr = flow["train"]; a = tr["arrays"]; tt = a["t"]; fl = lambda x: np.maximum(np.asarray(x, float), 1e-32)
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2)); ax = ax.ravel()
    keys = sorted((int(k) for k in tr["traces"] if int(k) > 0), key=int)
    want = [k for k in (1, 2, 5, 250, 1000, 2000, 4000, 6000, 8000) if str(k) in tr["traces"]]
    if len(want) < 4: want = keys[:: max(1, len(keys) // 7)]
    cm = plt.get_cmap("viridis")
    for i, k in enumerate(want):
        t_ = tr["traces"][str(k)]; ax[0].semilogy(np.arange(len(t_["f"])), fl(t_["f"]), "o-", ms=4, lw=1.4, color=cm(i / max(1, len(want) - 1)), label=rf"step {k} ($t$ = {t_['t']:.3g} s)")
    ax[0].set(xlabel="Newton iteration", ylabel=r"loss $\mathcal{L}=\langle r^2\rangle$ (discrete curl-momentum residual)", title="(a) Loss versus iteration, selected time steps"); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3, which="both")
    ax[1].semilogy(tt, fl(a["f0"]), color="0.55", lw=1.0, label="before the step (extrapolated predictor)"); ax[1].semilogy(tt, fl(a["f"]), color=C_SPIRAL, lw=1.4, label="after the step (converged)")
    ax[1].axvline(rec["t_inj"], color="k", ls=":", lw=1); ax[1].set(xlabel=r"simulation time $t$ [s]", ylabel=r"loss $\mathcal{L}$", title="(b) Loss per time step"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    ax[2].plot(tt, a["n_it"], color=C_SPIRAL, lw=1.0, label="Newton iterations per step"); ax[2].set(xlabel=r"simulation time $t$ [s]", ylabel="iterations per step", title="(c) Cost of the solve"); ax[2].grid(alpha=0.3)
    a2 = ax[2].twinx(); a2.plot(tt, np.cumsum(a["refact"]), color=C_SPIRAL2, lw=1.6, label="cumulative Cholesky factorisations"); a2.set_ylabel("cumulative factorisations")
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax[3].semilogy(tt, fl(a["rel"]), color=C_SPIRAL, lw=1.0); ax[3].axhline(CFG["tol_psi"], color="k", ls="--", lw=1, label=rf"tolerance {CFG['tol_psi']:g}")
    ax[3].set(xlabel=r"simulation time $t$ [s]", ylabel=r"relative update of $\mathbf{A}$ at termination", title="(d) Newton convergence criterion"); ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3, which="both")
    if abs(Re - 1000.0) < 1.0:
        tp = np.asarray(flow["t_psi"]); idx = np.unique(np.linspace(1, len(tp) - 1, 200).astype(int)); E = np.array([ghia_errors(flow["O"][i], N, Nz) for i in idx])
        ax[4].semilogy(tp[idx], E[:, 0], color=C_SPIRAL, lw=1.8, label=r"$u(y)$ on $x=L/2$, rel. $L_2$ vs " + GHIA_LAB); ax[4].semilogy(tp[idx], E[:, 1], color=C_SPIRAL2, lw=1.8, label=r"$v(x)$ on $y=L/2$, rel. $L_2$ vs " + GHIA_LAB)
        ax[4].semilogy(tp[idx], np.maximum(E[:, 2], 1e-3), color=C_SPIRAL3, lw=1.8, label=r"$\psi_{\min}$ (depth-mean) error vs " + GHIA_LAB)
        ax[4].set(xlabel=r"simulation time $t$ [s]", ylabel="error [%]", title=f"(e) Error vs 2D reference while the flow develops ({Z_WALL})"); ax[4].legend(fontsize=7.5); ax[4].grid(alpha=0.3, which="both")
    else:
        ax[4].axis("off"); ax[4].text(0.5, 0.5, "validation error vs Ghia et al. (1982)\nis defined for Re = 1000 only", ha="center")
    tau = rec["t"] - rec["t"][0]; ax[5].semilogy(tau[1:], fl(rec["res"][1:]), color=C_SPIRAL, lw=1.3, label="linear-solve residual of the scalar network")
    a3 = ax[5].twinx(); a3.plot(tau, rec["mass"] / rec["mass"][0] - 1, color=C_SPIRAL2, lw=1.5, label=r"mass drift $\int\phi\,dV$"); a3.set_ylabel("relative mass drift")
    ax[5].set(xlabel=r"time after injection [s]", ylabel="residual", title="(f) Scalar network: solve residual and conservation"); ax[5].grid(alpha=0.3, which="both")
    h1, l1 = ax[5].get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels(); ax[5].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.suptitle(f"{NAME}-3D: solver convergence ('training') history, Re = {Re:.0f}, {N}$\\times${N}$\\times${Nz} cells, {flow['stats']['features']:,} frozen-feature weights, no training data", fontsize=11)
    fig.tight_layout(); save_fig(fig, f"{folder}/Fig_training_convergence"); show(); plt.close(fig)
    say(f"\ntraining/convergence history: loss before step median {np.median(a['f0']):.2e} -> after step median {np.median(a['f']):.2e} (max {a['f'].max():.2e}); "
        f"Newton iterations/step mean {a['n_it'].mean():.2f} (max {int(a['n_it'].max())}); factorisations {int(a['refact'].sum())}")
    write_csv(f"{folder}/flow_training_history.csv", ["Time", "LossBefore", "LossAfter", "NewtonIts", "Factorisations", "RelUpdate"], [a["t"], a["f0"], a["f"], a["n_it"], a["refact"], a["rel"]])


def frame_png(fr, rec, flow, N, Nz, label, Re):
    t, ph3, o = fr; kk = mid_layers(Nz); phm = ph3[:, :, kk].mean(2); pha = ph3.mean(2)
    fig = plt.figure(figsize=(16.0, 7.4), dpi=CFG.get("gif_dpi", 100))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92], wspace=0.30, left=0.05, right=0.985, top=0.885, bottom=0.095)
    ax = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); rs = gs[0, 2].subgridspec(3, 1, hspace=0.62); ext = (0, L_ * 1e3, 0, L_ * 1e3); up = int(CFG.get("gif_up", 1))
    if up > 1:
        from scipy.ndimage import zoom as _zoom
        pm = np.clip(_zoom(phm, up, order=3, mode="nearest"), phm.min(), phm.max()); pa = np.clip(_zoom(pha, up, order=3, mode="nearest"), pha.min(), pha.max())
    else:
        pm, pa = phm, pha
    im = ax.imshow(pm.T, origin="lower", extent=ext, cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    imb = axb.imshow(pa.T, origin="lower", extent=ext, cmap="jet", vmin=0, vmax=max(pha.max(), 1e-6), interpolation="bilinear")
    psi = psi_depth_mean(o, N, Nz); gx = np.linspace(0, L_ * 1e3, N + 1); ax.contour(gx, gx, psi.T, levels=np.linspace(psi.min(), -1e-4, 9), colors="w", linewidths=0.5, alpha=0.6)
    m = rec["t"] <= t + 1e-9
    for a_ in (ax, axb):
        a_.plot(rec["cx"][m] * 1e3, rec["cy"][m] * 1e3, "k-", lw=1.4); a_.plot(rec["cx"][m][-1] * 1e3, rec["cy"][m][-1] * 1e3, "wo", mec="k", ms=6)
        a_.plot(rec["cx"][0] * 1e3, rec["cy"][0] * 1e3, "o", color="lime", mec="k", ms=5); a_.set_aspect("equal"); a_.grid(False); a_.set_xlabel(r"$x$ [mm]"); a_.set_ylabel(r"$y$ [mm]")
    ax.set_title(rf"(a) $\phi$ at mid-depth, fixed scale 0$-$1   ($t$ = {t:5.2f} s)", fontsize=12); axb.set_title(rf"(b) $\phi$ depth-averaged, auto-scaled 0$-${pha.max():.3f}", fontsize=12)
    for im_, a_ in ((im, ax), (imb, axb)):
        cb = fig.colorbar(im_, ax=a_, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=9); cb.set_label(r"$\phi$")
    a1 = fig.add_subplot(rs[0]); a1.plot(rec["t"], rec["mi"], color=C_SPIRAL, lw=1.8); a1.plot(t, rec["mi"][m][-1], "o", color=C_SPIRAL2); a1.set(xlim=(rec["t"][0], rec["t"][-1]), ylim=(-0.02, 1.0), ylabel="MI [$-$]", title="(c) Mixing index")
    a2 = fig.add_subplot(rs[1]); a2.plot(rec["t"], rec["pmax"], color=C_SPIRAL, lw=1.8); a2.plot(t, rec["pmax"][m][-1], "o", color=C_SPIRAL2); a2.set(xlim=(rec["t"][0], rec["t"][-1]), ylim=(0, 1.02), ylabel=r"$\phi_{\max}$ [$-$]", title="(d) Peak of $\\phi$")
    a3 = fig.add_subplot(rs[2]); a3.plot(flow["t"], flow["KE"] * 1e9, color=C_SPIRAL, lw=1.8); a3.axvline(rec["t"][0], color=C_SPIRAL2, ls="--", lw=1, label="injection")
    a3.plot(t, np.interp(t, flow["t"], flow["KE"]) * 1e9, "o", color=C_SPIRAL2); a3.legend(loc="lower right"); a3.set(xlabel=r"$t$ [s]", ylabel="KE [nJ]", title="(e) Kinetic energy")
    for a_ in (a1, a2, a3): a_.title.set_fontsize(11)
    fig.suptitle(f"{NAME}-3D  |  {label}  |  Re = {Re:.0f}  |  {N}$\\times${N}$\\times${Nz} cells, front/back {Z_WALL}", fontsize=12)
    fig.canvas.draw(); img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()); plt.close(fig)
    return img


def make_gif(rec, flow, N, Nz, label, Re, folder):
    if not rec["frames"]: return
    t0 = time.time(); imgs = [frame_png(fr, rec, flow, N, Nz, label, Re) for fr in rec["frames"]]
    d = int(1000 / CFG["gif_fps"]); hold = int(1000 * CFG.get("gif_hold_s", 1.5)); dur = [d] * len(imgs); dur[0] += hold; dur[-1] += hold          # slow motion + pause on the first and last frame
    imgs[0].save(f"{folder}/phi_mixing.gif", save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=False)
    step = max(1, len(imgs) // 6)
    for j, i in enumerate(range(0, len(imgs), step)): imgs[i].save(f"{folder}/phi_frame_{j:02d}_t{rec['frames'][i][0]:.2f}s.png")
    say(f"   GIF: {len(imgs)} frames at {CFG['gif_fps']} fps = {sum(dur) / 1000:.1f} s long for {rec['t'][-1] - rec['t'][0]:.1f} s of physical time ({time.time() - t0:.0f} s to render), {os.path.getsize(folder + '/phi_mixing.gif') / 2 ** 20:.1f} MB")


def outputs3(flow, rec, N, Nz, Re, label, folder, first=False):
    os.makedirs(folder, exist_ok=True)
    write_csv(f"{folder}/kinetic_energy.csv", ["Time", "KineticEnergy"], [flow["t"], flow["KE"]])
    write_csv(f"{folder}/mixing_index.csv", ["Time", "MixingIndex", "MeanPhi", "DMeanPhi"], [rec["t"], rec["mi"], rec["mean"], rec["dmean"]])
    write_csv(f"{folder}/center_of_gravity.csv", ["Time", "CoG_X", "CoG_Y", "CoG_Z"], [rec["t"], rec["cx"], rec["cy"], rec["cz"]])
    write_csv(f"{folder}/scalar_extra.csv", ["Time", "MixingIndexRaw", "PhiMax", "PhiMin", "TotalPhi", "LinearSolveResidual"], [rec["t"], rec["mi_raw"], rec["pmax"], rec["pmin"], rec["mass"], rec["res"]])
    say(f"\n[{label}] mixing index at the end ({rec['t'][-1]:.2f} s): {rec['mi'][-1]:.3f};  phi_max {rec['pmax'][-1]:.3f};  min phi {rec['pmin'].min():.3e};  CoG end ({rec['cx'][-1] * 1e3:.2f}, "
        f"{rec['cy'][-1] * 1e3:.2f}, {rec['cz'][-1] * 1e3:.2f}) mm;  total-phi drift {np.abs(rec['mass'] / rec['mass'][0] - 1).max():.1e}")
    fig_series(flow, rec, N, Nz, Re, label, folder)
    if first:
        fig_centrelines(flow, N, Nz, Re, folder); fig_vortices(flow, N, Nz, Re, folder); fig_vortex_vectors(flow, N, Nz, Re, folder); fig_3d_velocity(flow, N, Nz, Re, folder); fig_3d_streamlines(flow, N, Nz, Re, folder); fig_slices(flow, N, Nz, Re, folder); training_figure(flow, rec, N, Nz, Re, folder)
    fig_phi_sequence(rec, flow, N, Nz, Re, label, folder)
    make_gif(rec, flow, N, Nz, label, Re, folder)


# ---------------------------------------------------------------- cost report
def complexity_rows():
    rows = []
    for N_, Nz_ in ((20, 2), (30, 3), (40, 4), (50, 5), (64, 6), (100, 10)):
        s = grid_sizes(N_, Nz_); nA = s["nA"]; G = 8 * (s["nAx"] ** 2 + s["nAy"] ** 2 + s["nAz"] ** 2)
        rows.append(dict(N=N_, Nz=Nz_, cells=N_ * N_ * Nz_, nA=nA, G=G / 2 ** 30, KL64=8 * nA ** 2 / 2 ** 30, KL32=4 * nA ** 2 / 2 ** 30, chol=nA ** 3 / 3 / 1e9,
                         fits64=(G + 8 * nA ** 2 + TEMP_BYTES) < 0.85 * 14.56 * 2 ** 30, fits32=(G + 4 * nA ** 2 + TEMP_BYTES) < 0.85 * 14.56 * 2 ** 30))
    return rows


def write_metadata(hw, tot, st, N, Nz):
    import hashlib, datetime, scipy
    me = os.path.abspath(__file__) if "__file__" in globals() and os.path.exists(os.path.abspath(__file__)) else None
    cfg = {k: v for k, v in CFG.items() if isinstance(v, (int, float, str, bool, list, tuple, dict, type(None)))}
    meta = dict(created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), name=NAME + "-3D", script_sha256=(hashlib.sha256(open(me, "rb").read()).hexdigest() if me else None),
                numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__, hardware=hw, N=N, Nz=Nz, z_wall=Z_WALL, config=cfg, ansys_cost_supplied=ANSYS_COST, total=tot)
    json.dump(meta, open(f"{OUT}/run_metadata.json", "w"), indent=1, default=str)
    cuda = dev.type == "cuda"; rows = []
    for r in st:
        nm = r["stage"].replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        rows.append(f"{nm} & {r['wall_s']:.1f} & {r['cpu_s']:.1f} & {r['ram_peak_mb'] / 1024:.2f}" + (f" & {r.get('gpu_torch_peak_mb', float('nan')) / 1024:.2f}" if cuda else "") + r" \\")
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{Computational cost of " + NAME + f"-3D ({N}$\\times${N}$\\times${Nz} cells, " + (hw["gpu"] if cuda else "CPU only") + ").}\n\\begin{tabular}{l" + "r" * (4 if cuda else 3) + "}\n\\hline\n"
           "Stage & wall [s] & CPU [s] & RAM [GB]" + (" & GPU [GB]" if cuda else "") + r" \\ \hline" + "\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n")
    open(f"{OUT}/table_computational_cost.tex", "w").write(tex)


def cost_report(N, Nz, Re, runs):
    st = MON.stages; tot = MON.total(); hw = MON.hw; cuda = dev.type == "cuda"
    import pandas as pd
    pd.DataFrame(st).to_csv(f"{OUT}/computational_cost.csv", index=False)
    flow_st = [r for r in st if r.get("kind") == "flow"]; sc_st = [r for r in st if r.get("kind") == "scalar"]
    L = []; P = L.append; pm = predict_memory(N, Nz, CFG)
    P("=" * 112); P("COMPUTATIONAL COST OF THIS RUN (measured by the script)  -  SPIRAL-3D vs ANSYS Reference"); P("=" * 112)
    P(f"hardware : CPU {hw['cpu']} ({hw['cpu_threads']} threads), RAM {hw['ram_gb']:.1f} GB" + (f", GPU {hw['gpu']} ({hw['gpu_mem_gb']:.1f} GB, CUDA {hw['cuda']})" if cuda else ", NO GPU (CPU run)"))
    P(f"software : Python {hw['python']}, PyTorch {hw['torch']}, flow state float64, kernel factor {pm['factor_dtype']}, device = {hw['device']}")
    P(f"case     : {N} x {N} x {Nz} cells (h = {L_ / N * 1e3:.2f} mm in x,y; {DEPTH / Nz * 1e3:.2f} mm in z), front/back faces '{Z_WALL}', Re = {Re:.0f}, dt = {CFG['dt']} s")
    P("")
    hdr = f"{'stage':74s} {'wall [s]':>9} {'CPU [s]':>9} {'cores':>6} {'RAM peak [MB]':>14}" + (f" {'GPU mem torch [MB]':>19} {'GPU util mean/peak [%]':>23} {'GPU energy [Wh]':>16}" if cuda else "")
    P(hdr); P("-" * len(hdr))
    for r in st:
        line = f"{r['stage'][:74]:74s} {r['wall_s']:9.1f} {r['cpu_s']:9.1f} {r['cpu_cores_busy']:6.2f} {r['ram_peak_mb']:14.0f}"
        if cuda:
            ut = f"{r['gpu_util_mean_pct']:.0f} / {r['gpu_util_peak_pct']:.0f}" if "gpu_util_mean_pct" in r else "n/a"
            line += f" {r.get('gpu_torch_peak_mb', float('nan')):19.0f} {ut:>23} {r.get('gpu_energy_wh', float('nan')):16.3f}"
        P(line)
    P("-" * len(hdr))
    P(f"{'TOTAL (whole script incl. start-up, plots, GIFs)':74s} {tot['wall_s']:9.1f} {tot['cpu_s']:9.1f} {tot['cpu_s'] / max(tot['wall_s'], 1e-9):6.2f} {tot['ram_peak_mb']:14.0f}" + (f" {tot.get('gpu_torch_peak_mb', float('nan')):19.0f}" if cuda else ""))
    if cuda and "gpu_smi_mem_peak_mb" in tot: P(f"GPU memory seen by nvidia-smi (CUDA context + allocator cache, whole run): peak {tot['gpu_smi_mem_peak_mb']:.0f} MB")
    P("")
    P("SPIRAL-3D model size and cost per step")
    per_flow = per_sc = None
    if flow_st:
        f0 = flow_st[0]; per_flow = f0["wall_s"] / max(f0.get("steps", 1), 1)
        P(f"  flow network   : {f0.get('features', 0):,} output weights (3 outputs x {f0.get('features', 0) // 3:,} frozen tanh features), {f0.get('equations', 0):,} edge equations (curl of the momentum residual)")
        P(f"                   {f0.get('steps', 0):,} time steps in {f0['wall_s']:.0f} s = {per_flow:.3f} s per step (incl. kernel set-up); {f0.get('newton_its', 0):,} Newton iterations, {f0.get('factorisations', 0):,} Cholesky factorisations "
          f"({f0.get('newton_its', 0) / max(f0.get('steps', 1), 1):.2f} its/step, {f0.get('factorisations', 0) / max(f0.get('steps', 1), 1):.3f} factorisations/step)")
    else:
        P("  flow network   : loaded from a previous run (flow_history3d.npz); timing not repeated")
    if sc_st:
        s0 = sc_st[0]; per_sc = s0["wall_s"] / max(s0.get("steps", 1), 1)
        P(f"  scalar network : {s0.get('features', 0):,} features, {s0.get('equations', 0):,} cell equations; {s0.get('steps', 0):,} steps in {s0['wall_s']:.0f} s = {per_sc:.3f} s per step, {s0.get('factorisations', 0):,} factorisations")
    if per_flow and per_sc:
        n_f = int(f0["steps"]); n_s = int(runs[0][1]["n_steps"])
        P(f"  ANSYS-equivalent protocol ({n_f:,} flow steps + {n_s:,} scalar steps of dt = {CFG['dt']} s): {(per_flow * n_f + per_sc * n_s) / 60:.1f} min on this machine (flow {per_flow * n_f / 60:.1f} + scalar {per_sc * n_s / 60:.1f}; GIFs and plots excluded)")
    tot_sz = sum(os.path.getsize(os.path.join(d_, f_)) for d_, _, fs_ in os.walk(OUT) for f_ in fs_) / 2 ** 20
    P(f"  whole output folder (figures, GIFs, csv, flow history): {tot_sz:.0f} MB"); P("")
    P("ANSYS Reference (from 3D_Reference_CFD.pdf; wall time / cores / RAM are NOT stated there)")
    P("  model          : 3D slab 50 x 50 x 5 mm, hexahedral mesh, 5e-4 m elements (100 x 100 x 10), ~1.0e5 elements, 112,211 nodes; coupled pressure-velocity solver,")
    P("                   second-order implicit time stepping, 2nd-order upwind momentum + scalar; dt = 0.001 s, 4000 steps spin-up + 4000 steps with scalar; UDF mixing monitor (serial)")
    P("  unknowns/step  : about 5 x 1.0e5 = 5e5 coupled values (p, u, v, w + UDS-0), iterated to convergence in every time step")
    if ANSYS_COST: P("  YOUR numbers   : " + ", ".join(f"{k} = {v}" for k, v in ANSYS_COST.items()))
    else: P("  YOUR numbers   : not supplied.  Add them to get the time ratio, e.g.  S3_ANSYS_COST='{\"wall_h\": 2.5, \"cores\": 4, \"ram_gb\": 6, \"cpu\": \"your CPU\"}'")
    if "wall_h" in ANSYS_COST and per_flow and per_sc:
        t_sp = per_flow * n_f + per_sc * n_s; r = ANSYS_COST["wall_h"] * 3600 / t_sp
        P(f"  => ANSYS wall time {ANSYS_COST['wall_h']} h vs SPIRAL-3D (this machine, flow + scalar only) {t_sp / 3600:.2f} h: ANSYS / SPIRAL = {r:.2f} "
          f"({'SPIRAL faster' if r > 1 else 'ANSYS faster'}) - but SPIRAL here runs {N}x{N}x{Nz} = {N * N * Nz:,} cells, ANSYS 1.0e5 cells: compare accuracy at equal cost, not only time")
    P("")
    P("Resources and data needed to produce a result")
    rows = [("", "ANSYS Fluent Reference", "SPIRAL-3D (this work)"), ("software", "commercial licence, GUI/journal, UDF compiler", "free Python + PyTorch (Colab)"),
            ("geometry / mesh file", "CAD + hex mesh (~1.0e5 elements, 112,211 nodes)", f"none; {N} x {N} x {Nz} grid generated from 3 numbers"),
            ("training / reference data", "none (first-principles solver)", "none (physics only: no database, no labelled data, no pre-training)"),
            ("ANSYS result files (rcfd)", "-", "NOT needed to produce the result; used only to overlay / benchmark"),
            ("unknowns solved per step", "~5e5 (p, u, v, w, scalar)", f"{pm['features']:,} flow weights + {int(CFG['ratio'] * N * N * Nz):,} scalar weights"),
            ("hardware", ANSYS_COST.get("cpu", "CPU workstation / HPC (user to state)"), f"{hw.get('gpu', hw['cpu'])}"), ("deterministic", "yes", "yes (fixed seed, frozen random features)")]
    for a, b, c in rows: P(f"  {a:26s} | {b:52s} | {c}")
    P("")
    P("Complexity and memory model of the 3D flow network  (n_eq = number of interior edges = unknown A values, total features = ratio x n_eq, K and L dense)")
    P("  set-up     : G_c = Phi_c Phi_c^T per output  2 n_c^2 P flop (chunked, Phi never stored) ; stored: three G blocks (float64)")
    P("  factorise  : K = S G S^T + lam I (sparse x dense products), in-place Cholesky  n_eq^3 / 3 flop  (only when the chord is stale)")
    P("  per Newton : one triangular solve pair 2 n_eq^2 flop + sparse S^T + three G block products (n_c^2 each)  = O(n_eq^2)")
    P("  peak memory: 8 sum(n_c^2) (G) + b n_eq^2 (K factorised in place into L; b = 8 or 4 bytes) + ~1.2 GB buffers  ->  O((N^2 Nz)^2) = O(N^4 Nz^2)")
    P("  Cholesky time lower bounds on a T4: fp64 0.25 TFlop/s, fp32 8.1 TFlop/s (peak; real throughput is lower)")
    P(f"  {'N x N x Nz':>14} {'cells':>9} {'n_eq':>8} {'G [GB]':>7} {'K->L fp64':>10} {'K->L fp32':>10} {'Chol [GFlop]':>13} {'T4 fp64 >= [s]':>15} {'T4 fp32 >= [s]':>15}  fits a T4 (14.6 GB, 85 %)")
    for r in complexity_rows():
        fit = "fp64 + fp32" if r["fits64"] else ("fp32 factor only" if r["fits32"] else "NO (dense kernel too large)")
        P(f"  {str(r['N']) + 'x' + str(r['N']) + 'x' + str(r['Nz']):>14} {r['cells']:9d} {r['nA']:8d} {r['G']:7.2f} {r['KL64']:10.2f} {r['KL32']:10.2f} {r['chol']:13.1f} {r['chol'] * 1e9 / 0.25e12:15.1f} {r['chol'] * 1e9 / 8.1e12:15.1f}  {fit}"
          + ("   <- this run" if (r["N"], r["Nz"]) == (N, Nz) else "") + ("   <- the ANSYS mesh" if r["N"] == 100 else ""))
    P(f"  this run: predicted GPU memory flow {pm['flow'] / 2 ** 30:.2f} GB (G {pm['G'] / 2 ** 30:.2f} + K->L {pm['K_L'] / 2 ** 30:.2f} + buffers), scalar network {pm['scalar'] / 2 ** 30:.2f} GB (G, K, L)")
    if flow_st:
        f0 = flow_st[0]; P(f"  measured: peak process RAM {f0['ram_peak_mb']:.0f} MB" + (f", peak GPU memory (torch) {f0.get('gpu_torch_peak_mb', float('nan')):.0f} MB vs model {pm['flow'] / 2 ** 20:.0f} MB" if cuda else ""))
    P("")
    P("Caveats: wall times depend on the machine; T4 float64 throughput is 1/32 of float32; the dense-kernel formulation scales as O(n_eq^3) per factorisation, so the ANSYS mesh "
      "(100 x 100 x 10) is out of reach for it - SPIRAL-3D is a method study at 1-2 mm resolution, not a replacement for a 1e5-cell finite-volume run.")
    txt = "\n".join(L); open(f"{OUT}/computational_cost.txt", "w").write(txt + "\n"); print("\n" + txt, flush=True); REPORT.append(txt)
    write_metadata(hw, tot, st, N, Nz)
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 0.5 * len(st) + 3.4))
    names = [r["stage"].replace("post-processing: ", "post: ").replace("scalar network: ", "scalar: ")[:52] for r in st]; y = np.arange(len(st))[::-1]
    colr = {"flow": C_SPIRAL, "scalar": C_SPIRAL2, "post": "0.6"}; yl, nl = list(y), list(names)
    if "wall_h" in ANSYS_COST: yl.append(-1.3); nl.append(f"{REF} (Fluent, as supplied)")
    ax[0].barh(y, [max(r["wall_s"], 0.1) for r in st], color=[colr.get(r.get("kind"), "0.5") for r in st])
    if "wall_h" in ANSYS_COST: ax[0].barh([-1.3], [ANSYS_COST["wall_h"] * 3600], color="k", alpha=0.85)
    ax[0].set_xscale("log"); ax[0].set_yticks(yl); ax[0].set_yticklabels(nl, fontsize=7.5); ax[0].set(xlabel="wall-clock time [s] (log scale)", title="(a) Wall-clock time per stage"); ax[0].grid(axis="x", alpha=0.3)
    ax[1].barh(y + 0.2, [r["ram_peak_mb"] / 1024 for r in st], height=0.38, color="0.35", label="process RAM peak [GB]")
    if cuda: ax[1].barh(y - 0.2, [r.get("gpu_torch_peak_mb", 0) / 1024 for r in st], height=0.38, color="#e08a1e", label="GPU memory peak, torch [GB]")
    ax[1].set_yticks(yl); ax[1].set_yticklabels([]); ax[1].set_ylim(ax[0].get_ylim()); ax[1].set(xlabel="memory [GB]", title="(b) Peak memory per stage"); ax[1].legend(fontsize=8); ax[1].grid(axis="x", alpha=0.3)
    fig.suptitle(f"{NAME}-3D: computational cost, {N}$\\times${N}$\\times${Nz} cells, " + (hw["gpu"] if cuda else hw["cpu"]), fontsize=11); fig.tight_layout(); save_fig(fig, f"{OUT}/Fig_computational_cost"); show(); plt.close(fig)


# =============================================================== driver
def cache_key(N, Nz, Re):
    keys = ("ratio", "s_min", "s_max_cells", "dt", "n_start", "t_flow", "seed", "save_every_s", "max_it", "tol_psi", "lam_flow", "factor_dtype", "inject", "ke_window", "ke_tol", "t_flow_min", "t_flow_max")
    return json.dumps(dict({k: CFG[k] for k in keys}, N=N, Nz=Nz, Re=round(Re, 6), walls=WALLS, zr=ZR), sort_keys=True)


def get_flow(N, Nz, Re):
    path = f"{OUT}/flow_history3d.npz"; key = cache_key(N, Nz, Re)
    if os.path.exists(path):
        try:
            d = np.load(path, allow_pickle=False)
            if str(d["key"]) == key:
                print(f"flow: loaded from {path} (same configuration) - delete the file to recompute", flush=True)
                tr = json.loads(str(d["traces"])); arrays = {k[3:]: d[k] for k in d.files if k.startswith("tr_")}
                return dict(t=d["t"], KE=d["KE"], O=d["O"], t_psi=d["t_psi"], t_plateau=float(d["t_plateau"]), fn=None, maxdiv=float(d["maxdiv"]), plateau_found=bool(d["plateau_found"]) if "plateau_found" in d.files else True, N=N, Nz=Nz,
                            stats=json.loads(str(d["stats"])), train=dict(arrays=arrays, traces=tr), cached=True)
        except Exception as e:
            print("cache not usable:", e, flush=True)
    with MON.stage(f"flow network {N}x{N}x{Nz} (3D, front/back {Z_WALL})", kind="flow") as info:
        flow = run_flow(N, Nz, Re, CFG, WALLS, ckpt_path=f"{OUT}/flow_checkpoint.pt", key=key); info.update(flow["stats"])
    fn = flow["fn"]
    try:
        torch.save(dict(Z=fn.weights().cpu(), W=fn.W.cpu(), b=fn.b.cpu(), note="A_c(x) = sum_k Z[c,k] tanh(W[:,k].x + b[k]); x,y in [0,1] (units of L), z in [0, depth/L]"), f"{OUT}/flow_network_final.pt")
    except Exception as e:
        print("weights not saved:", e)
    np.savez(path, key=key, t=flow["t"], KE=flow["KE"], O=flow["O"], t_psi=flow["t_psi"], t_plateau=flow["t_plateau"], maxdiv=flow["maxdiv"], plateau_found=flow["plateau_found"], stats=json.dumps(flow["stats"]),
             traces=json.dumps(flow["train"]["traces"]), **{"tr_" + k: v for k, v in flow["train"]["arrays"].items()})
    fn.G = None; fn.fact = None; fn.Cur_t = None; fn.CurT_t = None; flow["fn"] = None
    if dev.type == "cuda": torch.cuda.empty_cache()
    return flow


def gpu_housekeeping():
    """Release memory left behind by an earlier failed run in the same Colab session (the traceback of the crashed cell keeps its tensors alive)."""
    if dev.type != "cuda": return
    import gc
    for a in ("last_traceback", "last_value", "last_type"):
        if hasattr(sys, a): setattr(sys, a, None)
    gc.collect(); torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(); used = (total - free) / 2 ** 30
    print(f"GPU memory in use before the run: {used:.2f} GB of {total / 2 ** 30:.2f} GB", flush=True)
    if used > 1.5:
        raise SystemExit(f"\nSTOP: {used:.1f} GB of the GPU are already occupied by something else in this session (most likely tensors kept alive by the traceback of a previous crashed run).\n"
                         f"Restart the runtime (Runtime -> Restart session), then run this script again as the first GPU cell.")


def main():
    gpu_housekeeping()
    Re = 1000.0 if RE_CASE == "nominal" else RE_TABLE
    N = int(CFG["N"]); Nz = int(CFG["Nz"] or max(2, round(N * ZR)))
    if N % 2: raise SystemExit("use an even N (the mid-lines then fall on grid lines)")
    pm = predict_memory(N, Nz, CFG)
    say("=" * 100)
    say(f"{NAME}-3D | {N} x {N} x {Nz} cells (h = {L_ / N * 1e3:.2f} mm), slab {L_ * 1e3:.0f} x {L_ * 1e3:.0f} x {DEPTH * 1e3:.0f} mm, front/back '{Z_WALL}', left/right no-slip, lid on top | Re = {Re:.0f} "
        f"(Re from the PDF table would be {RE_TABLE:.0f}) | Pe = {U_ * L_ / (CFG.get('D') or PHYS['D']):.0f}")
    if CFG["inject"] == "ke_flat":
        proto = (f"flow from rest until the kinetic energy is flat (|dKE|/KE < {CFG['ke_tol']:g} per {CFG['ke_window']:g} s window, not before {CFG['t_flow_min']:g} s, cap {CFG['t_flow_max']:g} s), "
                 f"then phi = 1 is injected in the circle and followed for {CFG['t_mix']:g} s")
    else:
        proto = (f"flow to {CFG['t_flow']} s, scalar injected at {CFG['t_inj']} s for {CFG['t_mix']} s" + (f" and again at {CFG['extra_inj']} s" if CFG["extra_inj"] else ""))
    say(f"unknowns: {pm['nA']:,} vector-potential edge values, {pm['features']:,} frozen tanh weights | dt = {CFG['dt']} s | {proto}")
    say(f"predicted GPU memory: flow {pm['flow'] / 2 ** 30:.2f} GB (G {pm['G'] / 2 ** 30:.2f} + K->L in place {pm['K_L'] / 2 ** 30:.2f} + buffers, factor {pm['factor_dtype']}), scalar {pm['scalar'] / 2 ** 30:.2f} GB | "
        f"Cholesky {pm['chol_flop'] / 1e9:.0f} GFlop per factorisation")
    if dev.type == "cuda":
        tot_mem = torch.cuda.get_device_properties(0).total_memory
        if max(pm["flow"], pm["scalar"]) * 1.12 > tot_mem:
            raise SystemExit(f"\nSTOP: predicted memory {max(pm['flow'], pm['scalar']) / 2 ** 30:.1f} GB does not fit in {tot_mem / 2 ** 30:.1f} GB. Use a smaller N (see the table in computational_cost.txt), "
                             f"or set S3_CFG='{{\"factor_dtype\": \"float32\"}}'.")
    flow = get_flow(N, Nz, Re)
    runs = []
    if CFG["inject"] == "ke_flat":
        plan = [(f"inject at KE flat (t = {flow['t_plateau']:.2f} s)", flow["t_plateau"], CFG["t_mix"], CFG["dt"] / CFG["scalar_sub"])]
    else:
        plan = [(f"ANSYS timeline (inject at {CFG['t_inj']:g} s)", CFG["t_inj"], CFG["t_mix"], CFG["dt"] / CFG["scalar_sub"])]
        for te in CFG["extra_inj"]:
            plan.append((f"inject at {te:g} s" + (" (flow frozen after %.1f s)" % flow["t"][-1] if te > flow["t"][-1] + 1e-9 else ""), te, CFG["t_mix_B"], CFG["dt_B"]))
    for i, (lab, te, tmix, dts_) in enumerate(plan):
        with MON.stage(f"scalar network: {lab}", kind="scalar") as info:
            rec = run_scalar(flow, N, Nz, CFG, te, tmix, "ABCDEFG"[i], dt_s=dts_, n_frames=CFG["n_frames"]); info.update(rec["stats"])
        runs.append((lab, rec, f"{OUT}/run_{'ABCDEFG'[i]}_inject_{te:.2f}s"))
    for i, (lab, rec, folder) in enumerate(runs):
        with MON.stage(f"post-processing: {lab}", kind="post"):
            outputs3(flow, rec, N, Nz, Re, lab, folder, first=(i == 0))
    say(f"\nflow: KE(t = {flow['t'][-1]:.1f} s) = {flow['KE'][-1] * 1e9:.2f} nJ;  " + (f"KE flat (window {CFG['ke_window']} s, tol {CFG['ke_tol']:g}) from t = {flow['t_plateau']:.2f} s;  " if flow.get("plateau_found", True) else f"KE not yet flat at the end (tol {CFG['ke_tol']:g});  ") +
        f"max |div u| = {flow['maxdiv']:.1e} (continuity holds identically)")
    cost_report(N, Nz, Re, runs)
    open(f"{OUT}/report.txt", "w").write("\n".join(REPORT) + "\n")
    zp = shutil.make_archive(f"{OUT}_all", "zip", OUT); print(f"\nEVERYTHING is in {OUT}  (zip: {zp})", flush=True)
    try:
        from google.colab import files; files.download(zp)
    except Exception:
        pass


if __name__ == "__main__":
    main()