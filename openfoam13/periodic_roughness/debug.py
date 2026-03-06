import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. LOAD DATA & SELECT PROFILE
# ==========================================================
data = np.load("data.npz")
nut, md, U, eta, h, u_star = (
    data["nut"],
    data["md"],
    data["U"],
    data["z"],
    data["h"],
    data["ustar"],
)

# Pick one profile in the middle of the dataset (e.g., inside a wake)
idx = len(nut) // 2

y_physical = eta[idx] * h[idx]
nut_true = nut[idx]
md_true = md[idx]  # This is the True LHS (Dnut/Dt)
U_true = U[idx]
h_val = h[idx]
ustar_val = u_star[idx]

# ==========================================================
# 2. ALGEBRAIC MODELS (Left Plot - Target: nu_t)
# ==========================================================
kappa = 0.41
nut_para = kappa * ustar_val * y_physical * (1.0 - y_physical / h_val)
nut_elder = np.full_like(y_physical, (kappa / 6.0) * ustar_val * h_val)

# ==========================================================
# 3. PDE SOURCE TERMS (Right Plot - Target: Material Derivative)
# ==========================================================
# Compute vertical gradient dU/dy (Assuming this is what your ML meant by 'dudx')
dU_dy = np.gradient(U_true, y_physical)

# --- A. Spalart-Allmaras ---
d_wall = np.maximum(y_physical, 1e-3 * h_val)  # Prevent divide-by-zero
cb1 = 0.1355
cw1 = 3.239

sa_production = cb1 * nut_true * np.abs(dU_dy)
sa_destruction = -cw1 * (nut_true / d_wall) ** 2
md_sa = sa_production + sa_destruction

# --- B. Your ML-Discovered Model ---
# Coefficients from your console output:
c1, c2, c3, c4 = 0.331363, 0.265394, -3.56790, 0.0117746

ml_term1 = c1 * (nut_true * np.abs(dU_dy))
ml_term2 = c2 * (nut_true * dU_dy)
ml_term3 = c3 * (nut_true**2 / h_val**2)
ml_term4 = c4 * (np.abs(U_true) * nut_true / h_val)

md_ml = ml_term1 + ml_term2 + ml_term3 + ml_term4

# ==========================================================
# 4. PLOTTING
# ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# --- Subplot 1: Algebraic State ---
ax1.plot(nut_true, y_physical, label="OpenFOAM (Truth)", color="black", linewidth=4)
ax1.plot(
    nut_para,
    y_physical,
    label="Parabolic Model",
    color="blue",
    linestyle="--",
    linewidth=3,
)
ax1.plot(
    nut_elder, y_physical, label="Elder Model", color="red", linestyle=":", linewidth=3
)

ax1.set_ylabel("Physical Depth $y$ (m)", fontsize=14)
ax1.set_xlabel("Eddy Viscosity $\\nu_t$ (m$^2$/s)", fontsize=14)
ax1.set_title("Algebraic Models vs. CFD Truth", fontsize=16)
ax1.legend(fontsize=12, loc="lower right")
ax1.grid(True, alpha=0.3)

# --- Subplot 2: PDE Transport ---
# Trim the first and last 2 points to avoid wall-boundary singularities ruining the X-axis scale
valid = slice(2, -2)

ax2.plot(
    md_true[valid],
    y_physical[valid],
    label="CFD Truth (True $D\\nu_t/Dt$)",
    color="black",
    linewidth=4,
)
ax2.plot(
    md_sa[valid],
    y_physical[valid],
    label="Spalart-Allmaras RHS",
    color="green",
    linestyle="-.",
    linewidth=3,
)
ax2.plot(
    md_ml[valid],
    y_physical[valid],
    label="ML-Discovered RHS",
    color="darkorange",
    linestyle="--",
    linewidth=3,
)

# Add a dashed line at x=0 to separate Production (Right) from Destruction (Left)
ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)

ax2.set_xlabel("Source Terms for $\\nu_t$ (m$^2$/s$^2$)", fontsize=14)
ax2.set_title("PDE Source Terms vs. True Material Derivative", fontsize=16)
ax2.legend(fontsize=12, loc="lower right")
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"Evaluating RANS Closures on Periodic Roughness\n($h={h_val:.2f}$m, $U_*={ustar_val:.3f}$m/s)",
    fontsize=18,
)
plt.tight_layout()
plt.show()
