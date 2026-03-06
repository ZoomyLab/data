import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from vtk_core import VTKOF, get_available_fields
from scipy.special import eval_sh_legendre  # Shifted Legendre evaluator
from sklearn.metrics import r2_score
from sklearn.linear_model import MultiTaskLasso


def extract_material_derivative_columns(
    d_2d_xy, n_moments=3, origin=(0, 0, 0), sweep_normal="x"
):
    # Pass uniform_n_eta=50 to get a stable grid
    c_list, u_list, _ = d_2d_xy.to_np(
        "U_x", normal=sweep_normal, origin=origin, use_point_data=True, uniform_n_eta=50
    )
    _, nut_list, _ = d_2d_xy.to_np(
        "nut", normal=sweep_normal, origin=origin, use_point_data=True, uniform_n_eta=50
    )
    _, adv_list, _ = d_2d_xy.to_np(
        "material_derivative",
        normal=sweep_normal,
        origin=origin,
        use_point_data=True,
        uniform_n_eta=50,
    )

    if not c_list:
        return np.empty((0, 1 + 5 * n_moments + 100))

    x_coords, a_moments, n_moments_list, M_moments = [], [], [], []
    nut_raw, md_raw = [], []
    h_list = []

    for centers, u_vals, nut_vals, adv_vals in zip(c_list, u_list, nut_list, adv_list):
        if len(centers) < n_moments + 2:
            continue

        h_phys = np.ptp(centers[:, 0])
        h_list.append(h_phys)

        eta = centers[:, 0]
        x_pos = centers[0, 1]

        ak = [
            np.trapezoid(u_vals * eval_sh_legendre(k, eta), eta)
            for k in range(n_moments)
        ]
        nk = [
            np.trapezoid(nut_vals * eval_sh_legendre(k, eta), eta)
            for k in range(n_moments)
        ]
        Mk = [
            np.trapezoid(adv_vals * eval_sh_legendre(k, eta), eta)
            for k in range(n_moments)
        ]

        x_coords.append(x_pos)
        a_moments.append(ak)
        n_moments_list.append(nk)
        M_moments.append(Mk)
        nut_raw.append(nut_vals)
        md_raw.append(adv_vals)

    # GUARD 1: No data found at all
    if len(x_coords) < 2:
        return np.empty((0, 1 + 5 * n_moments + 100))

    x_coords = np.array(x_coords)
    a_moments = np.array(a_moments)
    n_moments_list = np.array(n_moments_list)
    M_moments = np.array(M_moments)
    nut_raw = np.array(nut_raw)
    md_raw = np.array(md_raw)
    h_list = np.array(h_list)

    # Eliminate x-duplicates and sort
    _, unique_idx = np.unique(np.round(x_coords, decimals=6), return_index=True)
    unique_idx = np.sort(unique_idx)

    # GUARD 2: Not enough unique x-points to compute a gradient
    if len(unique_idx) < 2:
        return np.empty((0, 1 + 5 * n_moments + 100))

    x_coords = x_coords[unique_idx]
    a_moments = a_moments[unique_idx]
    n_moments_list = n_moments_list[unique_idx]
    M_moments = M_moments[unique_idx]
    nut_raw = nut_raw[unique_idx]
    md_raw = md_raw[unique_idx]
    h_list = h_list[unique_idx]

    try:
        # Now gradients are safe
        da_dx = np.gradient(a_moments, x_coords, axis=0)
        dn_dx = np.gradient(n_moments_list, x_coords, axis=0)
    except Exception as e:
        print(f"      [Warning] Gradient failed at x={x_coords[0]}: {e}")
        return np.empty((0, 1 + 5 * n_moments + 100))

    out_matrix = np.column_stack(
        (
            x_coords,
            h_list,
            a_moments,
            n_moments_list,
            da_dx,
            dn_dx,
            M_moments,
            nut_raw,
            md_raw,
        )
    )

    # Final safety: drop any row containing NaNs or Infs
    mask = np.all(np.isfinite(out_matrix), axis=1)
    return out_matrix[mask]


def build_material_dataset(
    sim, times, n_moments=3, save_path="material_derivative_data.npy"
):
    all_data = []

    for s in times:
        t_current = sim.get_time(s)
        s_prev = s - 1 if s > 0 else -1
        t_prev = sim.get_time(s_prev)
        dt = (
            t_current - t_prev
            if t_current > t_prev
            else sim.get_time(1) - sim.get_time(0)
        )

        print(
            f"Extracting material derivatives for time {t_current} (dt = {dt:.4f})..."
        )

        # 1. CURRENT FRAME
        d_3d = sim.get_time_step(s)
        if "nut" not in d_3d.point_data:
            d_3d = d_3d.cell_data_to_point_data()
        d_3d = d_3d.compute_derivative(scalars="nut", gradient=True)
        d_3d.point_data["material_derivative"] = np.sum(
            d_3d.point_data["U"] * d_3d.point_data["gradient"], axis=1
        )
        if "alpha.water" in get_available_fields(d_3d)[0]:
            d_3d = d_3d.threshold(0.9, scalars="alpha.water")
        data_curr = extract_material_derivative_columns(
            d_3d.slice(normal="z", origin=(0, 0, 0.05)),
            sweep_normal="x",
            n_moments=n_moments,
        )

        # 2. PREVIOUS FRAME
        d_3d_prev = sim.get_time_step(s_prev)
        if "nut" not in d_3d_prev.point_data:
            d_3d_prev = d_3d_prev.cell_data_to_point_data()
        d_3d_prev.point_data["material_derivative"] = np.zeros_like(
            d_3d_prev.point_data["nut"]
        )
        if "alpha.water" in get_available_fields(d_3d_prev)[0]:
            d_3d_prev = d_3d_prev.threshold(0.9, scalars="alpha.water")
        data_prev = extract_material_derivative_columns(
            d_3d_prev.slice(normal="z", origin=(0, 0, 0.05)),
            sweep_normal="x",
            n_moments=n_moments,
        )

        if data_curr.size == 0 or data_prev.size == 0:
            continue

        min_len = min(len(data_curr), len(data_prev))
        K = n_moments

        # Array Layout: x(1) | a(K) | n(K) | da(K) | dn(K) | M(K) | nut_raw(50) | md_raw(50)
        # Array Layout: x(0) | h(1) | a(K) | n(K) | da(K) | dn(K) | M(K) | nut_raw(50) | md_raw(50)
        idx_n = slice(2 + K, 2 + 2 * K)
        idx_M = slice(2 + 4 * K, 2 + 5 * K)
        idx_nut_raw = slice(2 + 5 * K, 2 + 5 * K + 50)
        idx_md_raw = slice(2 + 5 * K + 50, 2 + 5 * K + 100)

        # Add d/dt to Moments
        dn_dt = (data_curr[:min_len, idx_n] - data_prev[:min_len, idx_n]) / dt
        data_curr[:min_len, idx_M] += dn_dt

        # Add d/dt to Raw Profiles
        dnut_raw_dt = (
            data_curr[:min_len, idx_nut_raw] - data_prev[:min_len, idx_nut_raw]
        ) / dt
        data_curr[:min_len, idx_md_raw] += dnut_raw_dt

        all_data.append(data_curr[:min_len])

    final_dataset = np.vstack(all_data)
    np.save(save_path, final_dataset)
    return final_dataset


def get_moment_features(a_mom, n_mom, da_dx, dn_dx, h_list):
    """
    Highly enriched catalog including Cubic Terms for Production (n * a^2).
    """
    features = {}
    n_moms = a_mom.shape[1]
    inv_h = 1.0 / h_list
    inv_h2 = 1.0 / (h_list**2)  # Often appears in dissipation terms

    # 1. Linear & Intensity (Scale: L^2/T^2)
    for k in range(n_moms):
        features[f"a{k}*inv_h"] = a_mom[:, k] * inv_h
        features[f"n{k}*inv_h"] = n_mom[:, k] * inv_h
        features[f"|a{k}|*inv_h"] = np.abs(a_mom[:, k]) * inv_h

    # 2. Quadratic: (Production/Dissipation Proxies)
    for i in range(n_moms):
        for j in range(n_moms):
            # Velocity squared (Energy proxy)
            if i <= j:
                features[f"a{i}*a{j}*inv_h"] = a_mom[:, i] * a_mom[:, j] * inv_h

            # Linear Production/Coupling (u * nut)
            features[f"a{i}*n{j}*inv_h"] = a_mom[:, i] * n_mom[:, j] * inv_h

            # Dissipation Proxy (nut^2)
            if i <= j:
                features[f"n{i}*n{j}*inv_h"] = n_mom[:, i] * n_mom[:, j] * inv_h

    # 3. NEW: Cubic Terms (Classical RANS Production: nut * S^2)
    # We focus on nut * (a_i * a_j) / h
    for i in range(n_moms):  # Turbulence index
        for j in range(n_moms):  # Velocity 1
            for l in range(n_moms):  # Velocity 2
                if j <= l:
                    # Physical Meaning: n_i is the eddy viscosity, (a_j * a_l) is the strain squared
                    features[f"n{i}*a{j}*a{l}*inv_h"] = (
                        n_mom[:, i] * a_mom[:, j] * a_mom[:, l] * inv_h
                    )

    # 4. Horizontal Kinematics (Quasilinear Advection/Stretching)
    for i in range(n_moms):
        for j in range(n_moms):
            features[f"a{i} * dn{j}/dx"] = a_mom[:, i] * dn_dx[:, j]
            features[f"n{i} * da{j}/dx"] = n_mom[:, i] * da_dx[:, j]

    return features


def train_algebraic_pde(
    data_path="material_derivative_data.npy", n_moments=3, alpha=1e-4
):
    data = np.load(data_path)
    data = data[np.all(np.isfinite(data), axis=1)]

    x_coords = data[:, 0]
    h_list = data[:, 1]
    K = n_moments
    idx = 2
    a_mom = data[:, idx : idx + K]
    idx += K
    n_mom = data[:, idx : idx + K]
    idx += K
    da_dx = data[:, idx : idx + K]
    idx += K
    dn_dx = data[:, idx : idx + K]
    idx += K
    M_mom = data[:, idx : idx + K]

    # 1. Prepare Features
    features = get_moment_features(a_mom, n_mom, da_dx, dn_dx, h_list)
    feature_names = list(features.keys())
    X = np.column_stack(list(features.values()))

    # 2. Scale EVERYTHING (Features and Targets)
    # This is crucial for Multi-Task learning so M0 doesn't drown out M1/M2
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(M_mom)  # Scales all K moments to unit variance

    print(f"\n{'=' * 60}")
    print(f"DISCOVERING UNIFIED MOMENT SYSTEM (Multi-Task Lasso, K={K})")
    print(f"{'=' * 60}")

    # 3. Multi-Task Selection: Forces a shared sparsity pattern
    # If a feature is useful for ANY moment, it is kept for ALL moments.
    mt_lasso = MultiTaskLasso(alpha=alpha, fit_intercept=True, max_iter=100000)
    mt_lasso.fit(X_scaled, Y_scaled)

    # A feature survives if its norm across all tasks is significant
    mask = np.linalg.norm(mt_lasso.coef_, axis=0) > 1e-5
    surviving_names = [name for i, name in enumerate(feature_names) if mask[i]]

    trained_models = []
    masks = []

    # 4. Unbiased OLS Refinement for each row
    for k in range(K):
        y = M_mom[:, k]
        ols = LinearRegression(fit_intercept=True)

        if np.any(mask):
            X_survivors = X[:, mask]
            ols.fit(X_survivors, y)

            print(f"\n--- Unified Equation for M_{k} ---")
            terms = [f"{ols.intercept_:.5e}"]
            for name, coef in zip(surviving_names, ols.coef_):
                if abs(coef) > 1e-10:  # Clean up visual noise
                    terms.append(f"{coef:.5e} * [{name}]")
            print(f"M_{k}' = " + " +\n        ".join(terms))
            print(f"Moment R^2: {ols.score(X_survivors, y):.4f}")
        else:
            print(f"\n--- M_{k} ---")
            print(f"M_{k}' = 0.000")

        trained_models.append(ols)
        masks.append(mask)  # Shared mask for all

    return trained_models, masks, feature_names


def train_algebraic_pde(
    data_path="material_derivative_data.npy", n_moments=3, alpha=1e-4
):
    data = np.load(data_path)
    data = data[np.all(np.isfinite(data), axis=1)]

    x_coords = data[:, 0]
    h_list = data[:, 1]
    K = n_moments
    idx = 2
    a_mom = data[:, idx : idx + K]
    idx += K
    n_mom = data[:, idx : idx + K]
    idx += K
    da_dx = data[:, idx : idx + K]
    idx += K
    dn_dx = data[:, idx : idx + K]
    idx += K
    M_mom = data[:, idx : idx + K]

    # 1. Prepare Features
    features = get_moment_features(a_mom, n_mom, da_dx, dn_dx, h_list)
    feature_names = list(features.keys())
    X = np.column_stack(list(features.values()))

    # 2. Scale EVERYTHING (Features and Targets)
    # This is crucial for Multi-Task learning so M0 doesn't drown out M1/M2
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(M_mom)  # Scales all K moments to unit variance

    print(f"\n{'=' * 60}")
    print(f"DISCOVERING UNIFIED MOMENT SYSTEM (Multi-Task Lasso, K={K})")
    print(f"{'=' * 60}")

    # 3. Multi-Task Selection: Forces a shared sparsity pattern
    # If a feature is useful for ANY moment, it is kept for ALL moments.
    mt_lasso = MultiTaskLasso(alpha=alpha, fit_intercept=True, max_iter=100000)
    mt_lasso.fit(X_scaled, Y_scaled)

    # A feature survives if its norm across all tasks is significant
    mask = np.linalg.norm(mt_lasso.coef_, axis=0) > 1e-5
    surviving_names = [name for i, name in enumerate(feature_names) if mask[i]]

    trained_models = []
    masks = []

    # 4. Unbiased OLS Refinement for each row
    for k in range(K):
        y = M_mom[:, k]
        ols = LinearRegression(fit_intercept=True)

        if np.any(mask):
            X_survivors = X[:, mask]
            ols.fit(X_survivors, y)

            print(f"\n--- Unified Equation for M_{k} ---")
            terms = [f"{ols.intercept_:.5e}"]
            for name, coef in zip(surviving_names, ols.coef_):
                if abs(coef) > 1e-10:  # Clean up visual noise
                    terms.append(f"{coef:.5e} * [{name}]")
            print(f"M_{k}' = " + " +\n        ".join(terms))
            print(f"Moment R^2: {ols.score(X_survivors, y):.4f}")
        else:
            print(f"\n--- M_{k} ---")
            print(f"M_{k}' = 0.000")

        trained_models.append(ols)
        masks.append(mask)  # Shared mask for all

    return trained_models, masks, feature_names


def evaluate_3d_reconstruction(
    data_path, trained_models, masks, feature_names, n_moments=3, n_eta=50
):
    print(f"\n{'=' * 50}\nEVALUATING TRUE 3D PHYSICAL R^2 SCORES\n{'=' * 50}")

    data = np.load(data_path)
    data = data[np.all(np.isfinite(data), axis=1)]
    K = n_moments

    # 1. Precise Unpacking
    h_list = data[:, 1]
    a_mom = data[:, 2 : 2 + K]
    n_mom = data[:, 2 + K : 2 + 2 * K]
    da_dx = data[:, 2 + 2 * K : 2 + 3 * K]
    dn_dx = data[:, 2 + 3 * K : 2 + 4 * K]

    # Raw profiles are at the end: Skip x(1), h(1), a, n, da, dn, M (5*K)
    raw_offset = 2 + 5 * K
    true_nut_raw = data[:, raw_offset : raw_offset + n_eta]
    true_md_raw = data[:, raw_offset + n_eta : raw_offset + 2 * n_eta]

    # 2. Predict M_k using Features
    features = get_moment_features(a_mom, n_mom, da_dx, dn_dx, h_list)
    X_full = np.column_stack(list(features.values()))

    pred_M_mom = np.zeros((data.shape[0], K))
    for k in range(K):
        ols = trained_models[k]
        mask = masks[k]
        if np.any(mask):
            # Only predict if features were selected
            pred_M_mom[:, k] = ols.predict(X_full[:, mask])

    # 3. Reconstruct 3D profiles
    eta_grid = np.linspace(0, 1, n_eta)
    basis_funcs = np.array([eval_sh_legendre(k, eta_grid) for k in range(K)])

    # (Samples, K) @ (K, 50) -> (Samples, 50)
    pred_nut_raw = n_mom @ basis_funcs
    pred_md_raw = pred_M_mom @ basis_funcs

    # 4. Final Scoring
    r2_nut = r2_score(true_nut_raw.flatten(), pred_nut_raw.flatten())
    r2_md = r2_score(true_md_raw.flatten(), pred_md_raw.flatten())

    print(f"1. Projection Accuracy (Is K={K} enough for nut?): R^2 = {r2_nut:.4f}")
    print(f"2. Closure Accuracy (Does the ML PDE match 3D?):   R^2 = {r2_md:.4f}")


def train_3d_pde_operator(
    data_path="material_derivative_data.npy", n_moments=3, n_eta=50, alpha=1e-3
):
    data = np.load(data_path)
    data = data[np.all(np.isfinite(data), axis=1)]

    K = n_moments
    h_list = data[:, 1]

    raw_offset = 2 + 5 * K
    nut_profiles = data[:, raw_offset : raw_offset + n_eta]
    md_profiles = data[:, raw_offset + n_eta : raw_offset + 2 * n_eta]

    a_mom = data[:, 2 : 2 + K]
    da_dx_mom = data[:, 2 + 2 * K : 2 + 3 * K]
    eta_grid = np.linspace(0, 1, n_eta)
    basis_funcs = np.array([eval_sh_legendre(k, eta_grid) for k in range(K)])

    u_profiles = a_mom @ basis_funcs
    dudx_profiles = da_dx_mom @ basis_funcs

    # 3D vertical gradient dU/dz
    du_deta = np.gradient(u_profiles, eta_grid, axis=1)
    dudz_profiles = du_deta / h_list[:, np.newaxis]

    # --- THE FIX: Safe dh/dx computation across stacked timesteps ---
    unique_x, unique_idx = np.unique(data[:, 0], return_index=True)
    unique_h = h_list[unique_idx]

    # Compute gradient safely only on the unique spatial grid
    unique_dhdx = np.gradient(unique_h, unique_x)

    # Map back to the full dataset
    dh_dx = np.interp(data[:, 0], unique_x, unique_dhdx)
    dhdx_profiles = np.tile(dh_dx[:, np.newaxis], (1, n_eta))
    # ----------------------------------------------------------------

    inv_h = 1.0 / h_list[:, np.newaxis]
    Y_3d = md_profiles.flatten()

    # --- THE STRICTLY DIMENSIONALLY PURE CATALOG ---
    # Target (Y_3d) Units: [L^2 / T^2]

    features_3d = {
        # 1. Strain/Shear Production (Units: [L^2/T] * [1/T] = [L^2/T^2])
        # This is the classical Spalart-Allmaras production structure
        "nut * |dudz|": (nut_profiles * np.abs(dudz_profiles)).flatten(),
        "nut * |dudx|": (nut_profiles * np.abs(dudx_profiles)).flatten(),
        "nut * dudx": (
            nut_profiles * dudx_profiles
        ).flatten(),  # Keeps sign for accel/decel
        # 2. Destruction/Dissipation (Units: [L^4/T^2] / [L^2] = [L^2/T^2])
        "nut^2 / h^2": (nut_profiles**2 * inv_h**2).flatten(),
        # 3. Bulk Flow Interaction (Units: [L/T] * [L^2/T] / [L] = [L^2/T^2])
        "|U| * nut / h": (np.abs(u_profiles) * nut_profiles * inv_h).flatten(),
        "U * nut / h": (u_profiles * nut_profiles * inv_h).flatten(),
        # 4. Geometric / Parez Terms (Units: [L/T] * [L^2/T] * [-] / [L] = [L^2/T^2])
        "u * nut * dhdx / h": (
            u_profiles * nut_profiles * dhdx_profiles * inv_h
        ).flatten(),
        "|U| * nut * dhdx / h": (
            np.abs(u_profiles) * nut_profiles * dhdx_profiles * inv_h
        ).flatten(),
        # 5. Pure Velocity Terms (Units: [L/T]^2 = [L^2/T^2])
        # Sometimes mean kinetic energy acts directly as a wall-friction source
        "U^2": (u_profiles**2).flatten(),
        "U^2 * dhdx": (u_profiles**2 * dhdx_profiles).flatten(),
    }

    X_3d = np.column_stack(list(features_3d.values()))
    feature_names = list(features_3d.keys())

    # Ensure no NaNs from the edge cases
    valid_mask = np.all(np.isfinite(X_3d), axis=1) & np.isfinite(Y_3d)
    X_3d_clean = X_3d[valid_mask]
    Y_3d_clean = Y_3d[valid_mask]

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_3d_clean)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_3d_clean.reshape(-1, 1)).flatten()

    print(f"\n{'=' * 60}\nDISCOVERING DIMENSIONALLY PURE 3D SOURCE TERMS\n{'=' * 60}")

    # No intercept! The physics must explain the baseline.
    selector = Lasso(alpha=alpha, fit_intercept=False, max_iter=200000)
    selector.fit(X_scaled, Y_scaled)

    mask = np.abs(selector.coef_) > 1e-5
    if not np.any(mask):
        print(f"No terms selected at alpha={alpha}. Try decreasing alpha to 1e-4.")
        return None, None

    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_3d_clean[:, mask], Y_3d_clean)

    print("D(nut)/Dt = ")
    surviving_names = [feature_names[i] for i in range(len(mask)) if mask[i]]
    for name, coef in zip(surviving_names, ols.coef_):
        print(f"    {coef:+.5e} * [{name}]")

    print(f"\n3D Source Term R^2: {ols.score(X_3d_clean[:, mask], Y_3d_clean):.4f}")
    return ols, surviving_names
