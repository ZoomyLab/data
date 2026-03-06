from vtk_core import *
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from scipy.special import eval_sh_legendre  # Shifted Legendre evaluator
from sklearn.linear_model import LinearRegression


def extract_vertical_columns(d_2d_xy, n_moments=3, origin=(0, 0, 0), sweep_normal="x"):
    """
    Extracts columns and projects U_x onto shifted Legendre polynomials [0, 1].
    n_moments=3 will extract a0 (mean), a1, and a2.
    """
    c_list, u_list, _ = d_2d_xy.to_np(
        "U_x", normal=sweep_normal, origin=origin, use_point_data=True
    )
    _, nut_list, _ = d_2d_xy.to_np(
        "nut", normal=sweep_normal, origin=origin, use_point_data=True
    )

    extracted_data = []

    for centers, u_vals, nut_vals in zip(c_list, u_list, nut_list):
        if len(centers) < n_moments + 2:
            continue

        y_vals = centers[:, 1]
        y_vals = y_vals - np.min(y_vals)
        h = np.max(y_vals)
        if h < 1e-4:
            continue

        eta = y_vals / h

        # Compute Moments: a_k = integral(U * phi_k * d_eta)
        # Note: a0 will be proportional to U_bar
        moments = []
        for k in range(n_moments):
            phi_k = eval_sh_legendre(k, eta)
            # Projection via trapezoidal rule
            ak = np.trapezoid(u_vals * phi_k, eta)
            moments.append(ak)

        S_y = np.abs(np.gradient(u_vals, y_vals))

        # Prepare arrays for stacking
        h_arr = np.full_like(y_vals, h)
        moment_arrays = [np.full_like(y_vals, ak) for ak in moments]

        # Stack: [S, y, h, a0, a1, a2, ..., nut]
        col_data = np.column_stack((S_y, y_vals, h_arr, *moment_arrays, nut_vals))
        extracted_data.append(col_data)

    if not extracted_data:
        return np.empty((0, 3 + n_moments + 1))

    return np.vstack(extracted_data)


def get_features(p):
    # p = [S, z, h, a0, a1, a2, ...]
    S, z, h = p[0], p[1], p[2]
    moments = p[3:]

    eta = z / (h + 1e-8)
    d_surf = h - z
    parabola = z * d_surf / (h + 1e-8)

    features = {}

    # --- 1. The Internal Shapes (What nut(z) can look like) ---
    nut_shapes = {
        "z": z,
        "parabola": parabola,
        "phi1": 2 * eta - 1,  # Linear Shifted Legendre
        "phi2": 6 * eta**2 - 6 * eta + 1,  # Quadratic Shifted Legendre
    }

    # --- 2. The Coupling Logic ---
    for k, ak in enumerate(moments):
        abs_ak = np.abs(ak)
        for s_name, s_val in nut_shapes.items():
            # Intensity coupling (Always positive)
            features[f"|a{k}| * {s_name}"] = abs_ak * s_val

            # Directional coupling (Allows for asymmetry/skew)
            features[f"a{k} * {s_name}"] = ak * s_val

    # --- 3. Retain pure Local Shear for good measure ---
    features["S * z^2"] = S * z**2
    features["S * parabola"] = S * parabola

    return features


# def get_features(p):
#    # p = [S, z, h, a0, a1, a2, ...]
#    S, z, h = p[0], p[1], p[2]
#    moments = p[3:]  # List of moment arrays [a0, a1, a2, ...]
#
#    d_surf = h - z
#    eta = z / (h + 1e-8)
#
#    features = {}
#
#    # 1. Standard Parabolic Shapes
#    base_parabola = z * d_surf / (h + 1e-8)
#
#    # 2. Cross-multiply moments with shapes
#    # This allows the closure to be: nut = sum( c_ki * a_k * Shape_i )
#    for k, ak in enumerate(moments):
#        features[f"a{k} * z"] = ak * z
#        features[f"a{k} * z*(h-z)/h"] = ak * base_parabola
#        features[f"a{k} * z^2/h"] = ak * (z**2) / (h + 1e-8)
#
#    # 3. Add local shear terms
#    features["S * z^2"] = S * z**2
#    features["S * z*(h-z)"] = S * z * d_surf
#
#    return features
#
#    # def get_features(p):
#    S, z, h, U_bar, U_std = p[0], p[1], p[2], p[3], p[4]
#    eta = z / (h + 1e-8)  # Dimensionless height [0, 1]
#    d_surf = h - z
#
#    features = {
#        # --- 1. Linear/Parabolic Interaction ---
#        "U_bar * z * (h-z) / h": U_bar * z * d_surf / (h + 1e-8),
#        "U_std * z * (h-z) / h": U_std * z * d_surf / (h + 1e-8),
#        # --- 2. Higher Order Polynomials (for asymmetric profiles) ---
#        "U_std * z^2 * (h-z) / h^2": U_std * (z**2) * d_surf / (h**2 + 1e-8),
#        "U_bar * z * (h-z)^2 / h^2": U_bar * z * (d_surf**2) / (h**2 + 1e-8),
#        # --- 3. Shear-Length interaction ---
#        "S * z^2 * (h-z) / h": S * (z**2) * d_surf / (h + 1e-8),  # Cubic shear term
#        # --- 4. Scale interactions ---
#        "U_std * U_bar * z / h": (U_std * U_bar) * z / (h + 1e-8),
#        "(U_std^2 / U_bar) * z": (U_std**2 / (U_bar + 1e-8)) * z,
#    }
#    return features


# def get_features(p):
#    S, z, h, U_bar, U_std = p[0], p[1], p[2], p[3], p[4]
#    d_surf = h - z
#
#    features = {
#        # --- 1. Local Shear-driven ---
#        "S * z^2": S * z**2,
#        "S * z*(h-z)": S * z * d_surf,
#        # --- 2. Mean Bulk-Velocity driven ---
#        "U_bar * z*(h-z)/h": U_bar * z * d_surf / (h + 1e-8),
#        # --- 3. Variance-driven (The New Global Shear Scale) ---
#        "U_std * h": U_std
#        * h
#        * np.ones_like(z),  # Constant bulk mixing based on variance
#        "U_std * z": U_std * z,
#        "U_std * (h-z)": U_std * d_surf,
#        "U_std * z*(h-z)/h": U_std
#        * z
#        * d_surf
#        / (h + 1e-8),  # Parabola scaled by profile variance
#    }
#
#    return features


def train_sparse_closure(
    data_path="vam_closure_data.npy", feature_func=get_features, alpha_penalty=1e-4
):
    data = np.load(data_path)

    # 🚀 DYNAMIC UNPACKING
    # Column 0: S, Column 1: z, Column 2: h
    # Columns 3 to -2: Moments (a0, a1, a2...)
    # Last Column: nut
    S = data[:, 0]
    z = data[:, 1]
    h = data[:, 2]
    moments = [data[:, i] for i in range(3, data.shape[1] - 1)]
    nut_target = data[:, -1]

    # 1. Build the Feature Library
    features = feature_func([S, z, h, *moments])
    feature_names = list(features.keys())
    X = np.column_stack(list(features.values()))

    # 2. Solver Selection
    if alpha_penalty == 0:
        # Use LinearRegression for alpha=0 (Stable, Non-iterative)
        model = LinearRegression(positive=True, fit_intercept=True)
        X_train = X
    else:
        # Use Lasso for sparse discovery
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X)
        model = Lasso(
            alpha=alpha_penalty, positive=True, fit_intercept=True, max_iter=50000
        )

    model.fit(X_train, nut_target)

    # 3. Unscale coefficients
    if alpha_penalty > 0:
        actual_coeffs = model.coef_ / scaler.scale_
        # Adjust intercept for scaling
        intercept = model.intercept_ - np.sum(
            (model.coef_ * scaler.mean_) / scaler.scale_
        )
    else:
        actual_coeffs = model.coef_
        intercept = model.intercept_

    # 4. Output results
    print("\n--- Learned Closure Equation ---")
    terms = []

    # Check for constant background
    if abs(intercept) > 1e-8:
        terms.append(f"{intercept:.6e} (background)")

    for name, coef in zip(feature_names, actual_coeffs):
        if coef > 1e-10:
            terms.append(f"{coef:.6e} * [{name}]")

    if not terms:
        print(
            "Model found no correlation. Rebuild the dataset and check your feature units."
        )
    else:
        print("nut(z) = " + " +\n         ".join(terms))

    r2 = model.score(X_train, nut_target)
    print(f"\nModel R^2 Accuracy: {r2:.4f}")
    return model, feature_names, actual_coeffs


def build_training_dataset(sim, times, save_path="vam_closure_data.npy"):
    """
    Loops through timesteps, extracts columns, and saves the dataset.
    """
    all_data = []

    for s in times:
        print(f"Extracting time {sim.get_time(s)}...")
        d_3d = sim.get_time_step(s)

        # Threshold for water phase
        if "alpha.water" in get_available_fields(d_3d)[0]:
            d_3d = d_3d.threshold(0.9, scalars="alpha.water")

        # Create a vertical X-Y slice (assuming Y is the cross-stream normal)
        # Adjust origin/normal based on your actual 3D orientation!
        d_2d_xy = d_3d.slice(normal="z", origin=(0, 0, 0.05))

        data_matrix = extract_vertical_columns(d_2d_xy, sweep_normal="x", n_moments=5)
        if data_matrix.size > 0:
            all_data.append(data_matrix)

    final_dataset = np.vstack(all_data)
    np.save(save_path, final_dataset)
    print(f"Dataset saved to {save_path} with shape {final_dataset.shape}")
    return final_dataset
