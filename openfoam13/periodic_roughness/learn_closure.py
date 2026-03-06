from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from scipy.ndimage import gaussian_filter1d

import json
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d


def apply_shapes(base_features, eta):
    """Applies shape functions to a base dictionary of features."""
    features = {}
    for base_name, base_val in base_features.items():
        features[f"{base_name}"] = base_val
        features[f"[{base_name}] * eta"] = base_val * eta
        features[f"[{base_name}] * (1-eta)"] = base_val * (1.0 - eta)
    return features


def build_2eq_features(
    k,
    omega,
    nut,
    U,
    dU_dy,
    dU_dx,
    dk_dy,
    dk_dx,
    d2k_dy2,
    d2k_dx2,
    domega_dy,
    domega_dx,
    d2omega_dy2,
    d2omega_dx2,
    dnut_dy,
    dnut_dx,
    h,
    eta,
    ode=False,
):
    """Builds features for both k and omega. Drops dx terms if ode=True."""
    safe_omega = np.maximum(omega, 1e-10)

    # --- K-EQUATION BASE FEATURES ---
    k_base = {
        "nut * (dU_dy)^2": nut * (dU_dy**2),
        "+ nut * d2k_dy2": nut * d2k_dy2,
        "- nut * d2k_dy2": -(nut * d2k_dy2),
        "+ dnut_dy * dk_dy": dnut_dy * dk_dy,
        "- dnut_dy * dk_dy": -(dnut_dy * dk_dy),
        "- k * omega": -(k * safe_omega),  # Exact destruction for k-omega models
        # Horizontal Terms
        "nut * (dU_dx)^2": nut * (dU_dx**2),
        "+ nut * d2k_dx2": nut * d2k_dx2,
        "- nut * d2k_dx2": -(nut * d2k_dx2),
        "+ dnut_dx * dk_dx": dnut_dx * dk_dx,
        "- dnut_dx * dk_dx": -(dnut_dx * dk_dx),
    }

    # --- OMEGA-EQUATION BASE FEATURES ---
    omega_base = {
        "(dU_dy)^2": (dU_dy**2),  # Production for omega
        "+ nut * d2omega_dy2": nut * d2omega_dy2,
        "- nut * d2omega_dy2": -(nut * d2omega_dy2),
        "+ dnut_dy * domega_dy": dnut_dy * domega_dy,
        "- dnut_dy * domega_dy": -(dnut_dy * domega_dy),
        "+ (dk_dy * domega_dy) / omega": (dk_dy * domega_dy)
        / safe_omega,  # Classical SST Cross-diffusion!
        "- (dk_dy * domega_dy) / omega": -(dk_dy * domega_dy) / safe_omega,
        "- omega^2": -(omega**2),
        # Horizontal Terms
        "(dU_dx)^2": (dU_dx**2),
        "+ nut * d2omega_dx2": nut * d2omega_dx2,
        "- nut * d2omega_dx2": -(nut * d2omega_dx2),
        "+ dnut_dx * domega_dx": dnut_dx * domega_dx,
        "- dnut_dx * domega_dx": -(dnut_dx * domega_dx),
        "+ (dk_dx * domega_dx) / omega": (dk_dx * domega_dx) / safe_omega,
        "- (dk_dx * domega_dx) / omega": -(dk_dx * domega_dx) / safe_omega,
    }

    # Apply ODE filter if requested
    if ode:
        k_base = {key: val for key, val in k_base.items() if "_dx" not in key}
        omega_base = {key: val for key, val in omega_base.items() if "_dx" not in key}

    return apply_shapes(k_base, eta), apply_shapes(omega_base, eta)


def train_2eq_model(data_path="data.npz", alpha_k=1e-3, alpha_omega=1e-3, ode=False):
    data = np.load(data_path)
    trim = slice(1, -1)

    # 1. Unpack arrays and apply Gaussian smoothing
    def get_flat(key):
        return data[key][:, :, trim].flatten()

    def get_smooth(key):
        return gaussian_filter1d(data[key], sigma=1.0, axis=2)[:, :, trim].flatten()

    k, md_k = get_flat("k"), get_flat("md_k")
    omega, md_omega = get_flat("omega"), get_flat("md_omega")
    nut, U = get_flat("nut"), get_flat("U")

    dU_dy, dU_dx = get_flat("dU_dy"), get_flat("dU_dx")
    dk_dy, dk_dx = get_flat("dk_dy"), get_flat("dk_dx")
    domega_dy, domega_dx = get_flat("domega_dy"), get_flat("domega_dx")
    dnut_dy, dnut_dx = get_flat("dnut_dy"), get_flat("dnut_dx")

    d2k_dy2, d2k_dx2 = get_smooth("d2k_dy2"), get_smooth("d2k_dx2")
    d2omega_dy2, d2omega_dx2 = get_smooth("d2omega_dy2"), get_smooth("d2omega_dx2")

    n_times, n_stations, n_eta_trimmed = data["k"][:, :, trim].shape
    eta = np.tile(data["z"][trim], (n_times, n_stations, 1)).flatten()
    h = np.repeat(data["h"][:, :, np.newaxis], n_eta_trimmed, axis=2).flatten()

    # 2. Build feature libraries (with ODE constraint)
    k_features, omega_features = build_2eq_features(
        k,
        omega,
        nut,
        U,
        dU_dy,
        dU_dx,
        dk_dy,
        dk_dx,
        d2k_dy2,
        d2k_dx2,
        domega_dy,
        domega_dx,
        d2omega_dy2,
        d2omega_dx2,
        dnut_dy,
        dnut_dx,
        h,
        eta,
        ode=ode,
    )

    # 3. Helper to train SINDy pipeline
    def solve_sindy(features, target_md, target_name, alpha, outfile):
        X = np.column_stack(list(features.values()))
        Y = target_md
        X_scaled = StandardScaler().fit_transform(X)
        Y_scaled = StandardScaler().fit_transform(Y.reshape(-1, 1)).flatten()

        print(
            f"\n{'=' * 60}\nDISCOVERING {target_name.upper()} EQUATION {'(ODE)' if ode else '(PDE)'}\n{'=' * 60}"
        )

        selector = Lasso(
            alpha=alpha, fit_intercept=False, positive=True, max_iter=200000
        )
        selector.fit(X_scaled, Y_scaled)
        mask = selector.coef_ > 1e-5

        if not np.any(mask):
            print(f"FAILED: No terms selected for {target_name}. Try smaller alpha.")
            return

        ols = LinearRegression(fit_intercept=False, positive=True)
        ols.fit(X[:, mask], Y)

        print(f"D({target_name})/Dt = ")
        surviving_names = [
            list(features.keys())[i] for i in range(len(mask)) if mask[i]
        ]
        all_coeffs = {name: 0.0 for name in features.keys()}

        for name, coef in zip(surviving_names, ols.coef_):
            print(f"    {coef:+.5e} * {name}")
            all_coeffs[name] = coef

        with open(outfile, "w") as f:
            json.dump(all_coeffs, f, indent=4)
        print(f"R^2 ({target_name}): {ols.score(X[:, mask], Y):.4f}")

    # Solve for both
    solve_sindy(k_features, md_k, "k", alpha_k, "learned_k_coeffs.json")
    solve_sindy(
        omega_features, md_omega, "omega", alpha_omega, "learned_omega_coeffs.json"
    )


def build_feature_library(
    nut, U, dU_dy, dU_dx, dnut_dy, dnut_dx, d2nut_dy2, d2nut_dx2, h, eta
):
    """Constructs the exact physical features for both training and plotting."""
    inv_h = 1.0 / h

    base_features = {
        # --- POSITIVE PRODUCTION TERMS ---
        "nut * |dU_dy|": nut * np.abs(dU_dy),
        "nut * |dU_dx|": nut * np.abs(dU_dx),
        "h^2 * (dU_dy)^2": (h**2) * (dU_dy**2),
        "h^2 * (dU_dx)^2": (h**2) * (dU_dx**2),
        "U^2": U**2,
        # --- VERTICAL DIFFUSION TERMS ---
        "+ (dnut_dy)^2": dnut_dy**2,
        "- (dnut_dy)^2": -(dnut_dy**2),
        "+ nut * d2nut_dy2": nut * d2nut_dy2,
        "- nut * d2nut_dy2": -(nut * d2nut_dy2),
        # --- HORIZONTAL DIFFUSION TERMS ---
        "+ (dnut_dx)^2": dnut_dx**2,
        "- (dnut_dx)^2": -(dnut_dx**2),
        "+ nut * d2nut_dx2": nut * d2nut_dx2,
        "- nut * d2nut_dx2": -(nut * d2nut_dx2),
        # --- SIGNED PRODUCTION/ADVECTION TERMS ---
        "+ nut * dU_dx": nut * dU_dx,
        "- nut * dU_dx": -nut * dU_dx,
        # --- NEGATIVE DESTRUCTION TERMS ---
        "- nut^2 / h^2": -(nut**2) * (inv_h**2),
        "- |U| * nut / h": -np.abs(U) * nut * inv_h,
    }

    features = {}
    for base_name, base_val in base_features.items():
        # 1. Constant shape
        features[f"{base_name}"] = base_val
        # 2. Surface-weighted shape
        features[f"[{base_name}] * eta"] = base_val * eta
        # 3. Bed-weighted shape
        features[f"[{base_name}] * (1-eta)"] = base_val * (1.0 - eta)

    return features


def train_3d_pde_operator(data_path="data.npz", alpha=1e-3):
    data = np.load(data_path)

    # Trim the literal wall nodes (index 0 and 49) in the Z-dimension
    trim = slice(0, -1)

    # 1. Unpack 3D Arrays [Time, Space, Z]. We trim the 3rd axis!
    nut_3d = data["nut"][:, :, trim]
    n_times, n_stations, n_eta_trimmed = nut_3d.shape

    # 2. Extract Geometry and build 3D shapes
    eta_base = data["z"][trim]
    eta_3d = np.tile(eta_base, (n_times, n_stations, 1))

    h_2d = data["h"]
    h_3d = np.repeat(h_2d[:, :, np.newaxis], n_eta_trimmed, axis=2)

    # Broadcast dh_dx from 2D to 3D
    dh_dx_2d = data["dh_dx"]
    dh_dx_3d = np.repeat(dh_dx_2d[:, :, np.newaxis], n_eta_trimmed, axis=2)

    # 3. Flatten Everything for Scikit-Learn
    nut = nut_3d.flatten()
    md = data["md"][:, :, trim].flatten()
    U = data["U"][:, :, trim].flatten()

    dU_dy = data["dU_dy"][:, :, trim].flatten()
    dU_dx = data["dU_dx"][:, :, trim].flatten()
    d2U_dy2 = data["d2U_dy2"][:, :, trim].flatten()
    d2U_dx2 = data["d2U_dx2"][:, :, trim].flatten()

    # NO SCALING NEEDED! PyVista computed these in physical space (y)

    d2nut_dy2 = data["d2nut_dy2"]
    d2nut_dx2 = data["d2nut_dx2"]
    d2nut_dy2 = gaussian_filter1d(d2nut_dy2, sigma=0.4, axis=2)
    d2nut_dx2 = gaussian_filter1d(d2nut_dx2, sigma=0.4, axis=2)
    dnut_dy = data["dnut_dy"][:, :, trim].flatten()
    dnut_dx = data["dnut_dx"][:, :, trim].flatten()
    d2nut_dy2 = d2nut_dy2[:, :, trim].flatten()
    d2nut_dx2 = d2nut_dx2[:, :, trim].flatten()

    eta = eta_3d.flatten()
    h = h_3d.flatten()
    dh_dx = dh_dx_3d.flatten()

    inv_h = 1.0 / h

    features_3d = build_feature_library(
        nut, U, dU_dy, dU_dx, dnut_dy, dnut_dx, d2nut_dy2, d2nut_dx2, h, eta
    )

    X_3d = np.column_stack(list(features_3d.values()))
    Y_3d = md

    # Safety crash if any bad values somehow slipped in
    if not np.all(np.isfinite(X_3d)) or not np.all(np.isfinite(Y_3d)):
        raise ValueError("NaN or Inf encountered during feature construction!")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_3d)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_3d.reshape(-1, 1)).flatten()

    print(
        f"\n{'=' * 60}\nDISCOVERING PHYSICALLY CONSTRAINED SOURCE TERMS (WITH NATIVE DIFFUSION)\n{'=' * 60}"
    )

    # We use positive=True! The model is forced to combine terms additively.
    selector = Lasso(alpha=alpha, fit_intercept=False, positive=True, max_iter=200000)
    selector.fit(X_scaled, Y_scaled)

    mask = selector.coef_ > 1e-5
    if not np.any(mask):
        raise RuntimeError(
            "No terms selected. Try decreasing alpha (e.g., alpha=1e-4)."
        )

    ols = LinearRegression(fit_intercept=False, positive=True)
    ols.fit(X_3d[:, mask], Y_3d)

    print("D(nut)/Dt = ")
    feature_names = list(features_3d.keys())
    surviving_names = [feature_names[i] for i in range(len(mask)) if mask[i]]

    # 1. Create a dictionary of ALL features, defaulting to 0.0
    all_coeffs = {name: 0.0 for name in feature_names}

    # 2. Update the ones that survived the Lasso mask
    for name, coef in zip(surviving_names, ols.coef_):
        print(f"    {coef:+.5e} * {name}")
        # We save the exact raw coefficient. The physics signs (-/+) are
        # baked into the feature array itself!
        all_coeffs[name] = coef

    # 3. Dump to JSON
    coeff_path = "learned_coeffs.json"
    with open(coeff_path, "w") as f:
        json.dump(all_coeffs, f, indent=4)
    print(f"\nSaved all coefficients to {coeff_path}")

    print(f"3D Source Term R^2: {ols.score(X_3d[:, mask], Y_3d):.4f}")
    return ols, surviving_names, all_coeffs


def build_k_feature_library(
    k, nut, U, dU_dy, dU_dx, dk_dy, dk_dx, d2k_dy2, d2k_dx2, dnut_dy, dnut_dx, h, eta
):
    """Constructs the exact physical features for the k-equation PDE."""
    inv_h = 1.0 / h

    # Prevent divide-by-zero or complex numbers from tiny negative numerical noise
    safe_nut = np.maximum(nut, 1e-10)
    safe_k = np.maximum(k, 0.0)

    base_features = {
        # --- POSITIVE PRODUCTION TERMS ---
        "nut * (dU_dy)^2": nut * (dU_dy**2),
        "nut * (dU_dx)^2": nut * (dU_dx**2),
        # --- VERTICAL DIFFUSION (Laplacian + Cross-gradient) ---
        "+ nut * d2k_dy2": nut * d2k_dy2,
        "- nut * d2k_dy2": -(nut * d2k_dy2),
        "+ dnut_dy * dk_dy": dnut_dy * dk_dy,
        "- dnut_dy * dk_dy": -(dnut_dy * dk_dy),
        # --- HORIZONTAL DIFFUSION ---
        "+ nut * d2k_dx2": nut * d2k_dx2,
        "- nut * d2k_dx2": -(nut * d2k_dx2),
        "+ dnut_dx * dk_dx": dnut_dx * dk_dx,
        "- dnut_dx * dk_dx": -(dnut_dx * dk_dx),
        # --- NEGATIVE DESTRUCTION TERMS ---
        "- k^(3/2) / h": -(safe_k**1.5) * inv_h,
        "- k^2 / nut": -(k**2) / safe_nut,
        "- |U| * k / h": -np.abs(U) * k * inv_h,
    }

    features = {}
    for base_name, base_val in base_features.items():
        # 1. Constant shape
        features[f"{base_name}"] = base_val
        # 2. Surface-weighted shape
        features[f"[{base_name}] * eta"] = base_val * eta
        # 3. Bed-weighted shape
        features[f"[{base_name}] * (1-eta)"] = base_val * (1.0 - eta)

    return features


def train_k_pde_operator(data_path="data.npz", alpha=1e-3):
    data = np.load(data_path)

    # Trim the wall boundaries
    trim = slice(1, -1)  # Adjust to slice(10, -1) if you still have near-wall noise!

    # 1. Unpack 3D Arrays
    k_3d = data["k"][:, :, trim]
    n_times, n_stations, n_eta_trimmed = k_3d.shape

    # 2. Extract Geometry
    eta_base = data["z"][trim]
    eta_3d = np.tile(eta_base, (n_times, n_stations, 1))

    h_2d = data["h"]
    h_3d = np.repeat(h_2d[:, :, np.newaxis], n_eta_trimmed, axis=2)

    # 3. Flatten Base Variables
    k = k_3d.flatten()
    md_k = data["md_k"][:, :, trim].flatten()  # Target is now Dk/Dt
    nut = data["nut"][:, :, trim].flatten()
    U = data["U"][:, :, trim].flatten()

    dU_dy = data["dU_dy"][:, :, trim].flatten()
    dU_dx = data["dU_dx"][:, :, trim].flatten()
    dnut_dy = data["dnut_dy"][:, :, trim].flatten()
    dnut_dx = data["dnut_dx"][:, :, trim].flatten()

    # 4. Extract and Smooth k Derivatives
    d2k_dy2 = data["d2k_dy2"]
    d2k_dx2 = data["d2k_dx2"]
    # d2k_dy2 = gaussian_filter1d(d2k_dy2, sigma=0.4, axis=2)
    # d2k_dx2 = gaussian_filter1d(d2k_dx2, sigma=0.4, axis=2)

    dk_dy = data["dk_dy"][:, :, trim].flatten()
    dk_dx = data["dk_dx"][:, :, trim].flatten()
    d2k_dy2_flat = d2k_dy2[:, :, trim].flatten()
    d2k_dx2_flat = d2k_dx2[:, :, trim].flatten()

    eta = eta_3d.flatten()
    h = h_3d.flatten()

    # 5. Build Feature Matrix
    features_3d = build_k_feature_library(
        k,
        nut,
        U,
        dU_dy,
        dU_dx,
        dk_dy,
        dk_dx,
        d2k_dy2_flat,
        d2k_dx2_flat,
        dnut_dy,
        dnut_dx,
        h,
        eta,
    )

    X_3d = np.column_stack(list(features_3d.values()))
    Y_3d = md_k

    if not np.all(np.isfinite(X_3d)) or not np.all(np.isfinite(Y_3d)):
        raise ValueError("NaN or Inf encountered during k-feature construction!")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_3d)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_3d.reshape(-1, 1)).flatten()

    print(f"\n{'=' * 60}\nDISCOVERING K-EQUATION SOURCE TERMS\n{'=' * 60}")

    selector = Lasso(alpha=alpha, fit_intercept=False, positive=True, max_iter=200000)
    selector.fit(X_scaled, Y_scaled)

    mask = selector.coef_ > 1e-5
    if not np.any(mask):
        raise RuntimeError("No terms selected for k. Try decreasing alpha.")

    ols = LinearRegression(fit_intercept=False, positive=True)
    ols.fit(X_3d[:, mask], Y_3d)

    print("D(k)/Dt = ")
    feature_names = list(features_3d.keys())
    surviving_names = [feature_names[i] for i in range(len(mask)) if mask[i]]

    all_coeffs = {name: 0.0 for name in feature_names}

    for name, coef in zip(surviving_names, ols.coef_):
        print(f"    {coef:+.5e} * {name}")
        all_coeffs[name] = coef

    # Save to a separate JSON file so it doesn't overwrite your nut model!
    coeff_path = "learned_k_coeffs.json"
    with open(coeff_path, "w") as f:
        json.dump(all_coeffs, f, indent=4)
    print(f"\nSaved all k coefficients to {coeff_path}")

    print(f"3D Source Term R^2 (k-Equation): {ols.score(X_3d[:, mask], Y_3d):.4f}")
    return ols, surviving_names, all_coeffs
