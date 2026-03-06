import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json

# IMPORT BOTH FEATURE LIBRARIES HERE
from learn_closure import (
    build_feature_library,
    build_k_feature_library,
    build_2eq_features,
)


def evaluate_classical_models(data, t_slice=slice(None)):
    """
    Evaluates closures over a specified time slice.
    Defaults to slice(None) to evaluate all available time steps.
    """
    nut = data["nut"][t_slice]
    k = data["k"][t_slice]
    omega = data["omega"][t_slice]
    dU_dy = data["dU_dy"][t_slice]
    eta = data["z"]

    # Extract geometry/friction and dynamically add the Z-axis for broadcasting
    h_raw = data["h"][t_slice]
    ustar_raw = data["ustar"][t_slice]

    if h_raw.ndim == 2:  # Shape is [Time, Space]
        h_bcast = h_raw[:, :, np.newaxis]
        ustar_bcast = ustar_raw[:, :, np.newaxis]
    else:  # Shape is [Space] (if a single integer index was used)
        h_bcast = h_raw[:, np.newaxis]
        ustar_bcast = ustar_raw[:, np.newaxis]

    y_physical = h_bcast * eta

    # 1. Parabolic Model
    nut_para = 0.41 * ustar_bcast * h_bcast * eta * (1.0 - eta)

    # 2. Elder Model (Repeat along the Z-axis, which is always axis=-1)
    nut_elder_base = (0.41 / 6.0) * ustar_bcast * h_bcast
    nut_elder = np.repeat(nut_elder_base, nut.shape[-1], axis=-1)

    # 3. 1-Equation Practical Outcome (Prandtl)
    kappa, C_mu = 0.41, 0.548
    safe_1_minus_eta = np.maximum(1.0 - eta, 0.0)
    L = kappa * y_physical * np.sqrt(safe_1_minus_eta)
    nut_from_k_prandtl = C_mu * np.sqrt(np.maximum(k, 0.0)) * L

    # 4. Standard 2-Equation (Wilcox k-omega)
    safe_omega = np.maximum(omega, 1e-10)
    safe_k = np.maximum(k, 0.0)
    nut_from_k_omega = safe_k / safe_omega

    # --- 5. Menter's SST Limiter ---
    a1 = 0.31
    S = np.abs(dU_dy)
    sst_denominator = np.maximum(safe_omega, S / a1)
    nut_sst = safe_k / sst_denominator

    # Trim boundaries along the Z-axis (which is always the last axis)
    trim = slice(1, -1)

    # The '...' operator correctly handles both 2D and 3D arrays!
    nut_eval = nut[..., trim].flatten()
    nut_para_eval = nut_para[..., trim].flatten()
    nut_elder_eval = nut_elder[..., trim].flatten()
    nut_prandtl_eval = nut_from_k_prandtl[..., trim].flatten()
    nut_k_omega_eval = nut_from_k_omega[..., trim].flatten()
    nut_sst_eval = nut_sst[..., trim].flatten()

    print(
        f"\n{'=' * 60}\nEVALUATING ALGEBRAIC CLOSURES FOR nu_t (t_slice={t_slice})\n{'=' * 60}"
    )
    print(
        f"Data Sanity Check: Avg Depth (h) = {np.mean(h_raw):.4f} m, Avg U_* = {np.mean(ustar_raw):.4f} m/s"
    )
    print(f"[DEBUG] Max Physical nu_t:      {np.max(nut_eval):.5f}")
    print(f"[DEBUG] Max Raw k/omega nu_t:   {np.max(nut_k_omega_eval):.5f}")
    print(f"[DEBUG] Max SST Limited nu_t:   {np.max(nut_sst_eval):.5f}")

    print(f"1. Parabolic R^2              : {r2_score(nut_eval, nut_para_eval):+.4f}")
    print(f"2. Elder Model R^2            : {r2_score(nut_eval, nut_elder_eval):+.4f}")
    print(
        f"3. 1-Eq (Prandtl) R^2         : {r2_score(nut_eval, nut_prandtl_eval):+.4f}"
    )
    print(
        f"4. 2-Eq (Raw k-omega) R^2     : {r2_score(nut_eval, nut_k_omega_eval):+.4f}"
    )
    print(f"5. 2-Eq (SST Limited) R^2     : {r2_score(nut_eval, nut_sst_eval):+.4f}")
    print("=" * 60)


def _load_and_sum_features(coeff_file, features_1d, target_shape):
    """Helper to load JSON and multiply features."""
    md_ml = np.zeros(target_shape)
    try:
        with open(coeff_file, "r") as f:
            learned_coeffs = json.load(f)
        for name, coef in learned_coeffs.items():
            # If the feature exists in our dict and the coefficient isn't zero
            if name in features_1d and np.abs(coef) > 1e-8:
                md_ml += coef * features_1d[name]
    except FileNotFoundError:
        print(f"Warning: {coeff_file} not found. ML line will be flat.")
    return md_ml


def plot_algebraic_models(ax, data, t_idx=None, x_idx=None):
    """Plots the Algebraic closures (Parabolic & Elder) against the True nu_t on a given axis."""
    nut = data["nut"]
    eta = data["z"]
    h = data["h"]
    u_star = data["ustar"]

    # Pick the exact center of time and space if not provided
    if t_idx is None:
        t_idx = nut.shape[0] // 2
    if x_idx is None:
        x_idx = nut.shape[1] // 2

    # Extract pure 1D arrays for plotting
    y_physical = eta * h[t_idx, x_idx]
    nut_true = nut[t_idx, x_idx, :]
    h_val = h[t_idx, x_idx]
    ustar_val = u_star[t_idx, x_idx]

    # Calculate Algebraic Models
    kappa = 0.41
    nut_para = kappa * ustar_val * y_physical * (1.0 - y_physical / h_val)
    nut_elder = np.full_like(y_physical, (kappa / 6.0) * ustar_val * h_val)

    # Plotting
    ax.plot(nut_true, y_physical, label="OpenFOAM (Truth)", color="black", linewidth=4)
    ax.plot(
        nut_para,
        y_physical,
        label="Parabolic Model",
        color="blue",
        linestyle="--",
        linewidth=3,
    )
    ax.plot(
        nut_elder,
        y_physical,
        label="Elder Model",
        color="red",
        linestyle=":",
        linewidth=3,
    )

    ax.set_ylabel("Physical Depth $y$ (m)", fontsize=14)
    ax.set_xlabel("Eddy Viscosity $\\nu_t$ (m$^2$/s)", fontsize=14)
    ax.set_title("Algebraic Models vs. CFD Truth", fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_pde_models(ax, data, t_idx=None, x_idx=None, coeff_file="learned_coeffs.json"):
    """Plots the PDE source terms against True Material Derivative using dynamically loaded ML coefficients."""
    nut = data["nut"]
    md = data["md"]
    U = data["U"]
    dU_dy_all = data["dU_dy"]
    dU_dx_all = data["dU_dx"]
    d2nut_dy2_all = data["d2nut_dy2"]
    dnut_dy_all = data["dnut_dy"]
    eta = data["z"]
    h = data["h"]

    if t_idx is None:
        t_idx = nut.shape[0] // 2
    if x_idx is None:
        x_idx = nut.shape[1] // 2

    # Extract 1D arrays for plotting
    y_physical = eta * h[t_idx, x_idx]
    nut_true = nut[t_idx, x_idx, :]
    md_true = md[t_idx, x_idx, :]
    U_true = U[t_idx, x_idx, :]
    dU_dy = dU_dy_all[t_idx, x_idx, :]
    dU_dx = dU_dx_all[t_idx, x_idx, :]
    d2nut_dy2 = d2nut_dy2_all[t_idx, x_idx, :]
    dnut_dy = dnut_dy_all[t_idx, x_idx, :]
    h_val = h[t_idx, x_idx]

    dnut_dx = data["dnut_dx"][t_idx, x_idx, :]
    d2nut_dx2 = data["d2nut_dx2"][t_idx, x_idx, :]

    # --- A. Spalart-Allmaras ---
    d_wall = np.maximum(y_physical, 1e-3 * h_val)
    sa_production = 0.1355 * nut_true * np.abs(dU_dy)
    sa_destruction = -3.239 * (nut_true / d_wall) ** 2
    md_sa = sa_production + sa_destruction

    features_1d = build_feature_library(
        nut_true,
        U_true,
        dU_dy,
        dU_dx,
        dnut_dy,
        dnut_dx,
        d2nut_dy2,
        d2nut_dx2,
        h_val,
        eta,
    )

    # 2. Load coefficients and sum the active features
    md_ml = np.zeros_like(md_true)
    try:
        with open(coeff_file, "r") as f:
            learned_coeffs = json.load(f)

        for name, coef in learned_coeffs.items():
            if np.abs(coef) > 1e-8:  # Add non-zero features
                md_ml += coef * features_1d[name]
    except FileNotFoundError:
        print(f"Warning: {coeff_file} not found. ML line will be flat.")

    # --- Plotting ---
    valid = slice(1, -1)  # Trim wall-boundary singularities

    ax.plot(
        md_true[valid],
        y_physical[valid],
        "*-",
        label="CFD Truth",
        color="black",
        linewidth=4,
    )
    ax.plot(
        md_sa[valid],
        y_physical[valid],
        label="Spalart-Allmaras RHS",
        color="green",
        linestyle="-.",
        linewidth=3,
    )
    ax.plot(
        md_ml[valid],
        y_physical[valid],
        label="ML-Discovered RHS",
        color="darkorange",
        linestyle="--",
        linewidth=3,
    )

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Source Terms for $\\nu_t$ (m$^2$/s$^2$)", fontsize=14)
    ax.set_title(
        "PDE Source Terms vs. True Material Derivative ($\\nu_t$)", fontsize=16
    )
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_k_pde_models(
    ax, data, t_idx=None, x_idx=None, coeff_file="learned_k_coeffs.json"
):
    """Plots the PDE source terms against True Material Derivative for k."""
    k_all = data["k"]
    md_k_all = data["md_k"]
    nut_all = data["nut"]
    U_all = data["U"]
    eta = data["z"]
    h = data["h"]

    if t_idx is None:
        t_idx = k_all.shape[0] // 2
    if x_idx is None:
        x_idx = k_all.shape[1] // 2

    # Extract 1D arrays
    y_physical = eta * h[t_idx, x_idx]
    h_val = h[t_idx, x_idx]

    k_true = k_all[t_idx, x_idx, :]
    md_k_true = md_k_all[t_idx, x_idx, :]
    nut_true = nut_all[t_idx, x_idx, :]
    U_true = U_all[t_idx, x_idx, :]

    dU_dy = data["dU_dy"][t_idx, x_idx, :]
    dU_dx = data["dU_dx"][t_idx, x_idx, :]
    dk_dy = data["dk_dy"][t_idx, x_idx, :]
    dk_dx = data["dk_dx"][t_idx, x_idx, :]
    d2k_dy2 = data["d2k_dy2"][t_idx, x_idx, :]
    d2k_dx2 = data["d2k_dx2"][t_idx, x_idx, :]
    dnut_dy = data["dnut_dy"][t_idx, x_idx, :]
    dnut_dx = data["dnut_dx"][t_idx, x_idx, :]

    # --- A. Classical Prandtl One-Equation Baseline ---
    kappa = 0.41
    Cd = 0.164

    # Calculate algebraic mixing length L = kappa * y * sqrt(1 - y/h)
    L = kappa * y_physical * np.sqrt(np.maximum(1.0 - eta, 1e-6))
    L = np.maximum(L, 1e-3 * h_val)  # Prevent div-by-zero at the wall

    prandtl_production = nut_true * (dU_dy**2)
    prandtl_destruction = -Cd * (np.maximum(k_true, 0) ** 1.5) / L
    md_prandtl = prandtl_production + prandtl_destruction

    # --- B. ML-Discovered Model ---
    features_1d = build_k_feature_library(
        k_true,
        nut_true,
        U_true,
        dU_dy,
        dU_dx,
        dk_dy,
        dk_dx,
        d2k_dy2,
        d2k_dx2,
        dnut_dy,
        dnut_dx,
        h_val,
        eta,
    )

    md_ml = np.zeros_like(md_k_true)
    try:
        with open(coeff_file, "r") as f:
            learned_coeffs = json.load(f)
        for name, coef in learned_coeffs.items():
            if np.abs(coef) > 1e-8:
                md_ml += coef * features_1d[name]
    except FileNotFoundError:
        print(f"Warning: {coeff_file} not found. ML line will be flat.")

    # --- Plotting ---
    valid = slice(1, -1)  # Trim literal wall boundaries

    ax.plot(
        md_k_true[valid],
        y_physical[valid],
        "*-",
        label="CFD Truth (Dk/Dt)",
        color="black",
        linewidth=4,
    )
    ax.plot(
        md_prandtl[valid],
        y_physical[valid],
        label="Prandtl 1-Eq RHS",
        color="green",
        linestyle="-.",
        linewidth=3,
    )
    ax.plot(
        md_ml[valid],
        y_physical[valid],
        label="ML-Discovered RHS",
        color="darkorange",
        linestyle="--",
        linewidth=3,
    )

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Source Terms for $k$ (m$^2$/s$^3$)", fontsize=14)
    ax.set_title("PDE Source Terms vs. True Material Derivative ($k$)", fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_2eq_models(ax, data, t_idx=None, x_idx=None):
    """Plots the actual eddy viscosity (nu_t) closures against the CFD Truth."""
    if t_idx is None:
        t_idx = data["nut"].shape[0] // 2
    if x_idx is None:
        x_idx = data["nut"].shape[1] // 2

    # Extract 1D slices
    eta = data["z"]
    h_val = data["h"][t_idx, x_idx]
    ustar_val = data["ustar"][t_idx, x_idx]
    y_physical = eta * h_val

    nut_true = data["nut"][t_idx, x_idx, :]
    k_true = data["k"][t_idx, x_idx, :]
    omega_true = data["omega"][t_idx, x_idx, :]

    # --- 1. Parabolic Baseline ---
    nut_para = 0.41 * ustar_val * y_physical * (1.0 - eta)

    # --- 2. Prandtl 1-Equation Baseline ---
    kappa, C_mu = 0.41, 0.548
    safe_1_minus_eta = np.maximum(1.0 - eta, 0.0)
    L = kappa * y_physical * np.sqrt(safe_1_minus_eta)
    nut_prandtl = C_mu * np.sqrt(np.maximum(k_true, 0.0)) * L

    # --- 3. Wilcox k-omega Baseline ---
    safe_omega = np.maximum(omega_true, 1e-10)
    nut_k_omega = np.maximum(k_true, 0.0) / safe_omega

    # --- Plotting ---
    valid = slice(1, -1)  # Trim literal wall boundaries for a clean plot

    ax.plot(
        nut_true[valid],
        y_physical[valid],
        "*-",
        label="CFD Truth ($\\nu_t$)",
        color="black",
        linewidth=4,
    )
    ax.plot(
        nut_para[valid],
        y_physical[valid],
        label="Parabolic Model",
        color="blue",
        linestyle=":",
        linewidth=3,
    )
    ax.plot(
        nut_prandtl[valid],
        y_physical[valid],
        label="1-Eq (Prandtl) $\\nu_t$",
        color="green",
        linestyle="-.",
        linewidth=3,
    )
    ax.plot(
        nut_k_omega[valid],
        y_physical[valid],
        label="2-Eq (k-$\\omega$) $\\nu_t$",
        color="darkorange",
        linestyle="--",
        linewidth=3,
    )

    ax.set_xlabel("Eddy Viscosity $\\nu_t$ (m$^2$/s)", fontsize=14)
    ax.set_ylabel("Physical Depth $y$ (m)", fontsize=14)
    ax.set_title("Eddy Viscosity $\\nu_t$: Classical Closures vs. Truth", fontsize=16)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
