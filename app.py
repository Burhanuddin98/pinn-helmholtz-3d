# app.py — Streamlit PINN for 3D Helmholtz (Rigid/Neumann or Slightly Absorbing/Robin)
# + 3D viz + probes + robust training/logging
# Python 3.10+; numpy 1.26+; torch 2.2+ (CPU ok); deepxde 1.8.4+; streamlit; plotly
# Run: streamlit run app.py

import os, io, time, pathlib, contextlib
os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import torch as bkd  # noqa: F401 (ensures PyTorch backend import)
import torch
import plotly.graph_objects as go

# ---------------- paths / session state ----------------
OUTDIR = pathlib.Path.cwd() / "runs"
OUTDIR.mkdir(exist_ok=True)
def run_tag() -> str: return time.strftime("%Y%m%d-%H%M%S")

if "log" not in st.session_state: st.session_state["log"] = ""
if "model" not in st.session_state: st.session_state["model"] = None
if "last_tag" not in st.session_state: st.session_state["last_tag"] = ""

# ---------------- determinism + stable autodiff ----------------
def seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dde.config.set_random_seed(seed)

seed_all(0)
# IMPORTANT: use forward-mode to avoid reverse-mode Hessian cache bugs

# ===================== UI =====================
st.set_page_config(page_title="PINN Helmholtz 3D (Room Modes)", layout="wide")
st.title("PINN Helmholtz 3D — Room Sound Map (Helmholtz PDE)")

colA, colB = st.columns([2, 1], gap="large")

# A live console that we can refresh mid-run
console_placeholder = colA.empty()
with colA:
    console_placeholder.text_area("Logs", st.session_state["log"], height=260, disabled=True)
    row1 = st.columns([1, 1, 2])
    with row1[0]:
        train_btn = st.button("Train PINN")
    with row1[1]:
        render3d_btn = st.button("Render 3D from last model")
    plot_placeholder = st.empty()
    info_placeholder = st.empty()
    vol_placeholder = st.empty()
    dl_placeholder = st.empty()

with colB:
    st.subheader("Room & Sound")
    c  = st.number_input("Speed of sound c [m/s]", 300.0, 380.0, 343.0, 1.0)
    Lx = st.number_input("Room length Lx [m]",     1.0,  30.0, 5.0,   0.1)
    Ly = st.number_input("Room width  Ly [m]",     1.0,  30.0, 4.0,   0.1)
    Lz = st.number_input("Room height Lz [m]",     1.0,  15.0, 3.0,   0.1)

    f = st.number_input("Frequency f [Hz]", 10.0, 800.0, 100.0, 1.0)
    omega = 2.0 * np.pi * f
    k = float(omega / c)
    st.caption(f"k = {k:.4f} rad/m,  λ = {2*np.pi/k:.3f} m")

    st.markdown("**Source envelope (Gaussian)**")
    src_sigma  = st.number_input("Sigma [m]", 0.05, 1.00, 0.30, 0.01)
    src_gain   = st.number_input("Gain (amplitude)", 0.1, 50.0, 2.0, 0.1)

    st.markdown("**Carrier (helps reveal clean patterns)**")
    carrier_type = st.selectbox(
        "Carrier type",
        ["Mode-aligned cos(nₓπx/Lx)·cos(nᵧπy/Ly)·cos(n_zπz/Lz)", "Plane cos(kx)cos(ky)cos(kz)", "None"],
        index=0,
    )
    if carrier_type.startswith("Mode-aligned"):
        nx = int(st.number_input("nₓ", 0, 12, 1, 1))
        ny = int(st.number_input("nᵧ", 0, 12, 0, 1))
        nz = int(st.number_input("n_z", 0, 12, 0, 1))
        f_mode = (c / 2.0) * np.sqrt((nx / Lx) ** 2 + (ny / Ly) ** 2 + (nz / Lz) ** 2)
        st.caption(f"Eigenmode hint for ({nx},{ny},{nz}) ≈ {f_mode:.2f} Hz — set f near this.")

    st.markdown("**Source position**")
    src_x = st.slider("Source x [m]", 0.0, Lx, Lx/2.0, 0.05)
    src_y = st.slider("Source y [m]", 0.0, Ly, Ly/2.0, 0.05)
    src_z = st.slider("Source z [m]", 0.0, Lz, Lz/2.0, 0.05)

    st.subheader("Walls")
    wall_type = st.selectbox("Boundary type", ["Rigid (Neumann)", "Slightly absorbing (Robin)"], index=0)
    alpha = st.number_input("Absorption strength α [1/m]", 0.0, 5.0, 0.2, 0.05,
                            help="0 = rigid. Larger = more absorption.",
                            disabled=(wall_type=="Rigid (Neumann)"))

    st.subheader("Training")
    quick = st.toggle("Quick mode (fast VM-friendly)", value=True)

    # Super-fast smoke test: verifies end-to-end in <~1 min on CPU
    smoke_test = st.checkbox("Super-fast smoke test", value=False,
                             help="Use tiny network and few points to confirm logs/plots work.")

    if quick:
        width, depth = 64, 2
        adam_lr, adam_iters = 2e-3, 1200
        n_int, n_bnd = 3000, 1000
        display_every = 200
        resample_every = 0
        Nplot_default = 64
        N3D_default, N3D_max = 48, 72
    else:
        width, depth = 96, 3
        adam_lr, adam_iters = 1e-3, 5000
        n_int, n_bnd = 15000, 4000
        display_every = 500
        resample_every = 200
        Nplot_default = 96
        N3D_default, N3D_max = 64, 96

    # Override with smoke-test tiny settings
    if smoke_test:
        adam_lr = 3e-3
        adam_iters = 400
        n_int, n_bnd = 1200, 400
        width, depth = 32, 2
        display_every = 100
        resample_every = 0
        Nplot_default = 48
        N3D_default, N3D_max = 32, 64

    activation_key  = st.selectbox("Activation",  ["tanh", "sin", "relu"], index=0)
    initializer_key = st.selectbox("Initializer", ["Glorot uniform", "Glorot normal"], index=0)
    bc_weight = st.number_input("BC loss weight", 0.1, 50.0, 5.0, 0.1)

    run_lbfgs = st.toggle("Polish with L-BFGS (slower, cleaner)", value=False)

    slice_choice = st.selectbox("Slice to plot", ["z-mid", "y-mid", "x-mid"], index=0)
    Nplot = st.slider("Slice grid (N × N)", 48, 192, Nplot_default, 16)

    st.subheader("3D Visualization")
    do_3d_after_train = st.checkbox("Compute 3D field after training", value=False)
    N3D = st.slider("3D grid per axis (Nx=Ny=Nz)", 24, N3D_max, N3D_default, 8,
                    help="Total nodes = N^3. 64^3 ≈ 262k.")
    viz_type = st.selectbox("3D view", ["Isosurface", "Volume"], index=0)
    use_abs = st.checkbox("Plot |p| (magnitude) instead of Re{p}", value=False)

    # Robust viz controls
    sym_zero = st.checkbox("Symmetric range about 0", value=True,
                           help="Range set to [-A,+A] with A=max|p| (after optional de-meaning).")
    clip_mid_pct = st.slider("Keep middle % of values (clip tails)", 50, 99, 90, 1,
                             help="Clips outliers before setting viz range.")
    demean_for_viz = st.checkbox("Demean field for viz (subtract median)", value=True)

    if viz_type == "Isosurface":
        surf_count = st.slider("Isosurface count", 1, 5, 3, 1)
        iso_opacity = st.slider("Isosurface opacity", 0.10, 1.00, 0.60, 0.05)
        vol_surfaces = 12
        vol_opacity = 0.25
    else:
        vol_surfaces = st.slider("Volume surface count", 4, 24, 12, 2)
        vol_opacity = st.slider("Volume max opacity", 0.05, 1.00, 0.25, 0.05)
        surf_count = 3
        iso_opacity = 0.60

    st.subheader("Probes")
    seat_x = st.slider("Seat x [m]", 0.0, Lx, Lx*0.6, 0.05)
    seat_y = st.slider("Seat y [m]", 0.0, Ly, Ly*0.5, 0.05)
    seat_z = st.slider("Seat z [m]", 0.0, Lz, min(Lz*0.5, 1.2), 0.05, help="Ear height ~1.2 m if Lz≈2.4")

# ===================== helpers =====================
def predict_plane(model: dde.Model, Lx, Ly, Lz, const_axis: str, const_val: float, N: int = 96):
    xs = np.linspace(0.0, Lx, N); ys = np.linspace(0.0, Ly, N); zs = np.linspace(0.0, Lz, N)
    if const_axis == "z":
        X, Y = np.meshgrid(xs, ys, indexing="xy"); Z = np.full_like(X, const_val)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    elif const_axis == "y":
        X, Z = np.meshgrid(xs, zs, indexing="xy"); Y = np.full_like(X, const_val)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    elif const_axis == "x":
        Y, Z = np.meshgrid(ys, zs, indexing="xy"); X = np.full_like(Y, const_val)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    else:
        raise ValueError("const_axis must be 'x','y','z'")
    model.net.eval()
    with torch.no_grad():
        P = model.predict(pts).reshape(N, N)
    labels = {"z": ("x [m]", "y [m]"), "y": ("x [m]", "z [m]"), "x": ("y [m]", "z [m]")}[const_axis]
    if const_axis == "z": return X, Y, P, labels
    if const_axis == "y": return X, Z, P, labels
    return Y, Z, P, labels

def predict_grid_3d(model: dde.Model, Lx, Ly, Lz, Nx: int, Ny: int, Nz: int, batch: int = 131072):
    xg = np.linspace(0.0, Lx, Nx, dtype=np.float32)
    yg = np.linspace(0.0, Ly, Ny, dtype=np.float32)
    zg = np.linspace(0.0, Lz, Nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
    pts = np.stack([X.ravel(order="C"), Y.ravel(order="C"), Z.ravel(order="C")], axis=1)
    model.net.eval()
    vals = np.empty((pts.shape[0], 1), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, pts.shape[0], batch):
            end = min(start + batch, pts.shape[0])
            vals[start:end, :] = model.predict(pts[start:end, :])
    P = vals.reshape(Nx, Ny, Nz)
    return xg, yg, zg, P

def residual_rms(model, Lx, Ly, Lz, k, forcing_term, N=6000):
    device = next(model.net.parameters()).device
    X = (np.random.rand(N, 3).astype(np.float32) * np.array([Lx, Ly, Lz], dtype=np.float32))
    x_t = torch.tensor(X, dtype=torch.float32, requires_grad=True, device=device)
    y_t = model.net(x_t)
    dxx = dde.grad.hessian(y_t, x_t, i=0, j=0)
    dyy = dde.grad.hessian(y_t, x_t, i=0, j=1)
    dzz = dde.grad.hessian(y_t, x_t, i=0, j=2)
    r = dxx + dyy + dzz + (k**2) * y_t - forcing_term(x_t)
    pde_rms = float(torch.sqrt(torch.mean(r**2)).detach().cpu())

    M = N // 6
    faces = []
    faces += [np.c_[np.zeros((M,1)),        np.random.rand(M,1)*Ly, np.random.rand(M,1)*Lz]]
    faces += [np.c_[np.full((M,1), Lx),     np.random.rand(M,1)*Ly, np.random.rand(M,1)*Lz]]
    faces += [np.c_[np.random.rand(M,1)*Lx, np.zeros((M,1)),        np.random.rand(M,1)*Lz]]
    faces += [np.c_[np.random.rand(M,1)*Lx, np.full((M,1), Ly),     np.random.rand(M,1)*Lz]]
    faces += [np.c_[np.random.rand(M,1)*Lx, np.random.rand(M,1)*Ly, np.zeros((M,1))       ]]
    faces += [np.c_[np.random.rand(M,1)*Lx, np.random.rand(M,1)*Ly, np.full((M,1), Lz)    ]]
    XB = np.vstack(faces).astype(np.float32)

    xb = torch.tensor(XB, dtype=torch.float32, requires_grad=True, device=device)
    yb = model.net(xb)
    dpdx = dde.grad.jacobian(yb, xb, i=0, j=0)
    dpdy = dde.grad.jacobian(yb, xb, i=0, j=1)
    dpdz = dde.grad.jacobian(yb, xb, i=0, j=2)

    on_x = (xb[:, 0:1] == 0.0) | (xb[:, 0:1] == Lx)
    on_y = (xb[:, 1:2] == 0.0) | (xb[:, 1:2] == Ly)
    on_z = (xb[:, 2:3] == 0.0) | (xb[:, 2:3] == Lz)
    nb = torch.where(on_x, dpdx, torch.zeros_like(dpdx)) \
       + torch.where(on_y, dpdy, torch.zeros_like(dpdy)) \
       + torch.where(on_z, dpdz, torch.zeros_like(dpdz))

    bc_rms = float(torch.sqrt(torch.mean(nb**2)).detach().cpu())
    return pde_rms, bc_rms

def vtk_structured_points_bytes(xg, yg, zg, P, name="pressure"):
    Nx, Ny, Nz = P.shape
    dx = float(xg[1]-xg[0]) if Nx > 1 else 1.0
    dy = float(yg[1]-yg[0]) if Ny > 1 else 1.0
    dz = float(zg[1]-zg[0]) if Nz > 1 else 1.0
    header = [
        "# vtk DataFile Version 3.0", f"{name} field", "ASCII",
        "DATASET STRUCTURED_POINTS",
        f"DIMENSIONS {Nx} {Ny} {Nz}",
        "ORIGIN 0 0 0",
        f"SPACING {dx} {dy} {dz}",
        f"POINT_DATA {Nx*Ny*Nz}",
        "SCALARS pressure float 1",
        "LOOKUP_TABLE default",
    ]
    data_str = "\n".join(f"{v:.7e}" for v in P.ravel(order="F"))
    txt = "\n".join(header) + "\n" + data_str + "\n"
    return txt.encode("utf-8")

def render_plotly_3d(xg, yg, zg, P, mode="Isosurface", use_abs=False, surf_count=3,
                     iso_opacity=0.6, vol_surfaces=12, vol_opacity=0.25,
                     sym_zero=True, clip_mid_pct=90, demean=False):
    V = np.abs(P) if use_abs else P
    if demean: V = V - np.median(V)

    tail = (100 - clip_mid_pct) / 2.0
    lo_p = float(np.percentile(V, tail))
    hi_p = float(np.percentile(V, 100 - tail))
    Vc = np.clip(V, lo_p, hi_p)

    if sym_zero:
        rng = float(np.max(np.abs(Vc)))
        lo, hi = -rng, rng
    else:
        lo, hi = lo_p, hi_p
        if lo == hi: hi = lo + 1e-6

    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
    xr = X.ravel(order="F"); yr = Y.ravel(order="F"); zr = Z.ravel(order="F")
    vr = V.ravel(order="F")

    if mode == "Isosurface":
        fig = go.Figure(data=go.Isosurface(
            x=xr, y=yr, z=zr, value=vr,
            isomin=float(lo), isomax=float(hi),
            surface_count=int(surf_count),
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=float(iso_opacity),
            colorscale="Viridis", showscale=True,
        ))
    else:
        fig = go.Figure(data=go.Volume(
            x=xr, y=yr, z=zr, value=vr,
            isomin=float(lo), isomax=float(hi),
            surface_count=int(vol_surfaces),
            opacity=float(vol_opacity),
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale="Viridis", showscale=True,
        ))
    fig.update_layout(
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]", aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
        title="3D Field",
    )
    return fig

def compute_and_show_3d(model, Lx, Ly, Lz, N3D, viz_type, use_abs, surf_count,
                        iso_opacity, vol_surfaces, vol_opacity,
                        sym_zero, clip_mid_pct, demean_for_viz,
                        c, f, info_placeholder, vol_placeholder, dl_placeholder):
    with st.status("Computing 3D field…", expanded=False):
        xg, yg, zg, P = predict_grid_3d(model, Lx, Ly, Lz, N3D, N3D, N3D, batch=131072)

    # PPW (points-per-wavelength) check
    dx = Lx / (N3D - 1); dy = Ly / (N3D - 1); dz = Lz / (N3D - 1)
    lam = c / f
    ppw = min(lam / dx, lam / dy, lam / dz)
    if ppw < 10:
        info_placeholder.warning(
            f"Grid is coarse for f={f:.1f} Hz (min points/wavelength ≈ {ppw:.1f}). "
            "Increase N3D or lower f for cleaner modes."
        )

    fig3d = render_plotly_3d(
        xg, yg, zg, P, mode=viz_type, use_abs=use_abs,
        surf_count=surf_count, iso_opacity=iso_opacity,
        vol_surfaces=vol_surfaces, vol_opacity=vol_opacity,
        sym_zero=sym_zero, clip_mid_pct=clip_mid_pct, demean=demean_for_viz
    )
    vol_placeholder.plotly_chart(fig3d, use_container_width=True)

    tag = st.session_state.get("last_tag", run_tag())
    vtk_bytes = vtk_structured_points_bytes(xg, yg, zg, P, name="pressure")
    dl_placeholder.download_button(
        "Download 3D field (.vtk, Structured Points)",
        data=vtk_bytes,
        file_name=f"field3d-{tag}.vtk",
        mime="application/octet-stream",
    )
    buf_npz = io.BytesIO()
    np.savez(buf_npz, xg=xg, yg=yg, zg=zg, P=P, Lx=Lx, Ly=Ly, Lz=Lz, f=f, k=2*np.pi*f/c)
    dl_placeholder.download_button(
        "Download 3D field (.npz)",
        data=buf_npz.getvalue(),
        file_name=f"field3d-{tag}.npz",
        mime="application/octet-stream",
    )
    info_placeholder.info(f"3D grid: {N3D}×{N3D}×{N3D} nodes  |  min={P.min():.3e}, max={P.max():.3e}")

# ===================== training =====================
if train_btn:
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    geom = dde.geometry.Cuboid([0.0, 0.0, 0.0], [Lx, Ly, Lz])

    # forcing term (Gaussian at user-chosen source position with optional carrier)
    def forcing_term(x: torch.Tensor) -> torch.Tensor:
        xc = torch.tensor([src_x, src_y, src_z], dtype=x.dtype, device=x.device)
        r2 = torch.sum((x - xc) ** 2, dim=1, keepdim=True)
        gauss = torch.exp(-r2 / (2.0 * (src_sigma ** 2)))
        amp_t = torch.as_tensor(src_gain, dtype=x.dtype, device=x.device)

        if carrier_type.startswith("Mode-aligned"):
            Lx_t = torch.as_tensor(Lx, dtype=x.dtype, device=x.device)
            Ly_t = torch.as_tensor(Ly, dtype=x.dtype, device=x.device)
            Lz_t = torch.as_tensor(Lz, dtype=x.dtype, device=x.device)
            nx_t = torch.as_tensor(nx, dtype=x.dtype, device=x.device)
            ny_t = torch.as_tensor(ny, dtype=x.dtype, device=x.device)
            nz_t = torch.as_tensor(nz, dtype=x.dtype, device=x.device)
            carrier = (torch.cos(nx_t * np.pi * x[:, 0:1] / Lx_t) *
                       torch.cos(ny_t * np.pi * x[:, 1:2] / Ly_t) *
                       torch.cos(nz_t * np.pi * x[:, 2:3] / Lz_t))
            return amp_t * carrier * gauss

        if carrier_type.startswith("Plane"):
            k_t = torch.as_tensor(k, dtype=x.dtype, device=x.device)
            carrier = (torch.cos(k_t * x[:, 0:1]) *
                       torch.cos(k_t * x[:, 1:2]) *
                       torch.cos(k_t * x[:, 2:3]))
            return amp_t * carrier * gauss

        return amp_t * gauss  # None → pure Gaussian

    # PDE residual
    def pde_residual(x, y):
        dxx = dde.grad.hessian(y, x, i=0, j=0)
        dyy = dde.grad.hessian(y, x, i=0, j=1)
        dzz = dde.grad.hessian(y, x, i=0, j=2)
        lap = dxx + dyy + dzz
        return lap + (k ** 2) * y - forcing_term(x)

    # Unified boundary operator: Neumann (rigid) or Robin (slightly absorbing)
    def boundary_operator_bc(x, y, _):
        dpdx = dde.grad.jacobian(y, x, i=0, j=0)
        dpdy = dde.grad.jacobian(y, x, i=0, j=1)
        dpdz = dde.grad.jacobian(y, x, i=0, j=2)
        x0, y0, z0 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        zero = torch.zeros(1, dtype=x.dtype, device=x.device)
        Lx_t = torch.as_tensor(Lx, dtype=x.dtype, device=x.device)
        Ly_t = torch.as_tensor(Ly, dtype=x.dtype, device=x.device)
        Lz_t = torch.as_tensor(Lz, dtype=x.dtype, device=x.device)

        on_x0 = torch.isclose(x0, zero); on_xL = torch.isclose(x0, Lx_t)
        on_y0 = torch.isclose(y0, zero); on_yL = torch.isclose(y0, Ly_t)
        on_z0 = torch.isclose(z0, zero); on_zL = torch.isclose(z0, Lz_t)

        # outward normal derivative on each face
        dpdn = torch.where(on_x0, -dpdx, torch.zeros_like(dpdx)) + torch.where(on_xL, dpdx, torch.zeros_like(dpdx)) \
             + torch.where(on_y0, -dpdy, torch.zeros_like(dpdy)) + torch.where(on_yL, dpdy, torch.zeros_like(dpdy)) \
             + torch.where(on_z0, -dpdz, torch.zeros_like(dpdz)) + torch.where(on_zL, dpdz, torch.zeros_like(dpdz))

        if wall_type == "Rigid (Neumann)":
            return dpdn
        else:
            alpha_t = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
            return dpdn + alpha_t * y

    bc = dde.icbc.OperatorBC(geom, boundary_operator_bc, on_boundary=lambda x, on_b: on_b)

    data = dde.data.PDE(
        geom, pde_residual, [bc],
        num_domain=int(n_int), num_boundary=int(n_bnd),
        train_distribution="pseudo",
    )

    # model
    layer_sizes = [3] + [width] * depth + [1]
    try:
        net = dde.nn.FNN(layer_sizes, activation_key, initializer_key)
    except TypeError:
        net = dde.nn.FNN(layer_sizes, activation=activation_key, kernel_initializer=initializer_key)
    model = dde.Model(data, net)

    tag = run_tag()
    ckpt = dde.callbacks.ModelCheckpoint(
        str(OUTDIR / f"ckpt-{tag}.pt"),
        save_better_only=False,
        period=int(max(50, display_every))
    )
    callbacks_adam = [ckpt]
    if resample_every:
        callbacks_adam.append(dde.callbacks.PDEPointResampler(period=int(resample_every)))

    # -------- Stage 1: Adam warm-up --------
    # ensure enough steps even if quick: use max of user/quick/smoke-test and a floor
    adam_iters_warmup = max(800 if quick else 3000, int(adam_iters))

    buf = io.StringIO()
    try:
        with st.status("Training (Adam)…", expanded=False) as st_status:
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(buf):
                model.compile("adam", lr=adam_lr, loss_weights=[1.0, bc_weight])
                losshist_adam, _ = model.train(
                    iterations=int(adam_iters_warmup),
                    callbacks=callbacks_adam,
                    display_every=100,
                )
            dt = time.perf_counter() - t0
            st_status.update(label=f"Adam complete in {dt:.1f}s", state="complete")
    except Exception as e:
        st.exception(e)
    finally:
        st.session_state["log"] += buf.getvalue()
        # refresh console immediately
        console_placeholder.text_area("Logs", st.session_state["log"], height=260, disabled=True)

    # -------- Stage 2: L-BFGS polish (optional) --------
    if run_lbfgs:
        buf2 = io.StringIO()
        try:
            with st.status("Polishing (L-BFGS)…", expanded=False) as st_status:
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(buf2):
                    model.compile("L-BFGS")
                    losshist_bfgs, _ = model.train()  # run to convergence
                dt = time.perf_counter() - t0
                st_status.update(label=f"L-BFGS complete in {dt:.1f}s", state="complete")
        except Exception as e:
            st.warning("L-BFGS failed; keeping Adam result.")
            st.exception(e)
        finally:
            st.session_state["log"] += buf2.getvalue()
            console_placeholder.text_area("Logs", st.session_state["log"], height=260, disabled=True)

    # residual metrics
    try:
        pde_rms, bc_rms = residual_rms(model, Lx, Ly, Lz, k, forcing_term, N=6000)
        st.session_state["log"] += f"\nPDE residual RMS: {pde_rms:.3e}   BC residual RMS: {bc_rms:.3e}\n"
        console_placeholder.text_area("Logs", st.session_state["log"], height=260, disabled=True)
    except Exception as e:
        st.warning("Residual metrics failed.")
        st.exception(e)

    # 2D slice + probes
    which = slice_choice.split("-")[0]
    const_val = {"z": Lz/2.0, "y": Ly/2.0, "x": Lx/2.0}[which]
    A, B, P2, labels = predict_plane(model, Lx, Ly, Lz, which, const_val, N=Nplot)
    fig, ax = plt.subplots(figsize=(8, 4))
    pc = ax.pcolormesh(A, B, P2, shading="auto")
    fig.colorbar(pc, ax=ax, label="Re{p}")
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_title(f"{which}-mid slice")

    # Overlay source + seat if on the shown plane
    if which == "z" and abs(seat_z - const_val) < 1e-6:
        ax.plot(src_x, src_y, "wx", markersize=6, markeredgecolor="k")
        ax.plot(seat_x, seat_y, "wo", markersize=6, markeredgecolor="k")
    elif which == "y" and abs(seat_y - const_val) < 1e-6:
        ax.plot(src_x, src_z, "wx", markersize=6, markeredgecolor="k")
        ax.plot(seat_x, seat_z, "wo", markersize=6, markeredgecolor="k")
    elif which == "x" and abs(seat_x - const_val) < 1e-6:
        ax.plot(src_y, src_z, "wx", markersize=6, markeredgecolor="k")
        ax.plot(seat_y, seat_z, "wo", markersize=6, markeredgecolor="k")

    plot_placeholder.pyplot(fig); plt.close(fig)

    # Seat probe readout
    with torch.no_grad():
        p_seat = float(model.predict(np.array([[seat_x, seat_y, seat_z]], dtype=np.float32))[0, 0])
    info_placeholder.info(f"Seat probe @ ({seat_x:.2f},{seat_y:.2f},{seat_z:.2f}) m → p ≈ {p_seat:.3e}")

    # Save slice + download
    tag = run_tag()
    out_npz = OUTDIR / f"slice_{which}mid-{tag}.npz"
    np.savez(out_npz, A=A, B=B, P=P2, Lx=Lx, Ly=Ly, Lz=Lz, f=f, k=k,
             src=(src_x, src_y, src_z), seat=(seat_x, seat_y, seat_z),
             wall_type=wall_type, alpha=alpha)
    buf_npz = io.BytesIO()
    np.savez(buf_npz, A=A, B=B, P=P2, Lx=Lx, Ly=Ly, Lz=Lz, f=f, k=k,
             src=(src_x, src_y, src_z), seat=(seat_x, seat_y, seat_z),
             wall_type=wall_type, alpha=alpha)
    dl_placeholder.download_button(
        "Download slice (.npz)", data=buf_npz.getvalue(),
        file_name=out_npz.name, mime="application/octet-stream",
    )

    st.session_state["model"] = model
    st.session_state["last_tag"] = tag

    if do_3d_after_train:
        compute_and_show_3d(
            model, Lx, Ly, Lz, N3D, viz_type, use_abs, surf_count, iso_opacity,
            vol_surfaces, vol_opacity, sym_zero, clip_mid_pct, demean_for_viz,
            c, f, info_placeholder, vol_placeholder, dl_placeholder
        )

# -------- Render 3D from last model (no retrain) --------
if render3d_btn:
    if st.session_state.get("model", None) is None:
        st.warning("Train the PINN first — no model in session.")
    else:
        model = st.session_state["model"]
        # Recompute seat probe on demand too
        with torch.no_grad():
            p_seat = float(model.predict(np.array([[seat_x, seat_y, seat_z]], dtype=np.float32))[0, 0])
        info_placeholder.info(f"(Re)computed seat probe → p ≈ {p_seat:.3e}")
        compute_and_show_3d(
            model, Lx, Ly, Lz, N3D, viz_type, use_abs, surf_count, iso_opacity,
            vol_surfaces, vol_opacity, sym_zero, clip_mid_pct, demean_for_viz,
            c, f, info_placeholder, vol_placeholder, dl_placeholder
        )
