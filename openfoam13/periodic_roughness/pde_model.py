import numpy as np
from scipy.special import eval_sh_legendre
from vtk_core import get_available_fields

try:
    import pysindy as ps
except ImportError:
    print("PySINDy not found. Please run: pip install pysindy")


def build_training_dataset_sindy(
    sim, times, n_eta=50, n_moments=3, save_path="sindy_pde_data.npz"
):
    """
    Extracts data, computes Legendre moments, and maps it onto a structured,
    dimensionless vertical grid (eta = [0,1]).
    """
    eta_grid = np.linspace(0, 1, n_eta)
    all_nut = []
    all_moments = []  # We store moments now, not 'u'
    valid_times = []

    for s in times:
        t_phys = sim.get_time(s)
        print(f"[SINDy Build] Extracting structured grid for time {t_phys}...")

        d_3d = sim.get_time_step(s)
        if "alpha.water" in get_available_fields(d_3d)[0]:
            d_3d = d_3d.threshold(0.9, scalars="alpha.water")
        d_2d_xy = d_3d.slice(normal="z", origin=(0, 0, 0.05))

        c_list, u_list, _ = d_2d_xy.to_np(
            "U_x", normal="x", origin=(0, 0, 0), use_point_data=True
        )
        _, nut_list, _ = d_2d_xy.to_np(
            "nut", normal="x", origin=(0, 0, 0), use_point_data=True
        )

        nut_interp_cols = []
        moment_cols = []

        # Map every column onto the exact same grid
        for centers, u_vals, nut_vals in zip(c_list, u_list, nut_list):
            if len(centers) < n_moments + 2:
                continue

            y_vals = centers[:, 1]
            y_vals = y_vals - np.min(y_vals)
            h = np.max(y_vals)
            if h < 1e-4:
                continue

            eta = y_vals / h

            # Interpolate nut to uniform grid
            nut_interp_cols.append(np.interp(eta_grid, eta, nut_vals))

            # Calculate moments
            ak_list = []
            for k in range(n_moments):
                phi_k = eval_sh_legendre(k, eta)
                ak_list.append(np.trapezoid(u_vals * phi_k, eta))
            moment_cols.append(ak_list)

        if len(nut_interp_cols) > 0:
            # Spatially average to get a clean 1D (vertical) Spatiotemporal PDE
            all_nut.append(np.mean(nut_interp_cols, axis=0))
            all_moments.append(np.mean(moment_cols, axis=0))  # Shape: (n_moments,)
            valid_times.append(t_phys)

    nut_matrix = np.vstack(all_nut)  # Shape: (times, eta)
    moments_matrix = np.vstack(all_moments)  # Shape: (times, moments)
    t_array = np.array(valid_times)

    np.savez(save_path, nut=nut_matrix, moments=moments_matrix, eta=eta_grid, t=t_array)
    print(
        f"SINDy PDE structured dataset saved to {save_path}. Shape: {nut_matrix.shape}"
    )


def train_sindy_pde(data_path="sindy_pde_data.npz", threshold=0.01):
    """
    Discovers a PDE using PySINDy: d(nut)/dt = F(nut, d_nut/d_eta, d2_nut/d_eta2, a0, a1...)
    """
    print("\n--- Initializing PySINDy PDE Discovery ---")

    # 1. Load structured data
    data = np.load(data_path)
    nut = data["nut"]  # Shape (time, eta)
    moments = data["moments"]  # Shape (time, n_moments)
    eta = data["eta"]
    t = data["t"]

    n_times, n_eta = nut.shape
    n_moments = moments.shape[1]

    # We must combine nut and moments into a single 3D array for SINDy: (times, eta, features)
    X_data = np.zeros((n_times, n_eta, 1 + n_moments))
    X_data[:, :, 0] = nut
    for k in range(n_moments):
        # Broadcast the scalar moment across the entire depth for this timestep
        X_data[:, :, k + 1] = np.repeat(moments[:, k][:, np.newaxis], n_eta, axis=1)

    # Naming the features
    feature_names = ["nut"] + [f"a{k}" for k in range(n_moments)]

    # 2. Define the Custom Base Library for VAM (PySINDy 2.x Syntax)
    custom_functions = [lambda x: x, lambda x: np.abs(x)]
    custom_names = [lambda x: x, lambda x: f"|{x}|"]

    # Step A: Build the base algebraic library
    base_lib = ps.CustomLibrary(
        library_functions=custom_functions, function_names=custom_names
    )

    # Step B: Combine with Spatial Derivatives (1st and 2nd order)
    pde_lib = ps.PDELibrary(
        function_library=base_lib,  # <--- This is the PySINDy 2.x Fix!
        derivative_order=2,  # Discover up to 2nd derivative (Diffusion)
        spatial_grid=eta,  # The spatial domain
        include_bias=True,  # Allow constant background mixing
    )

    # 3. Define Optimizer
    # STLSQ is the robust backbone of SINDy
    optimizer = ps.STLSQ(threshold=threshold, alpha=1e-5, max_iter=5000)

    # 4. Train
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)

    # Reshape for SINDy format requirement: (spatial_dim, time, features)
    X_fit = np.transpose(X_data, (1, 0, 2))

    print("Fitting SINDy model... (Computing numerical derivatives)")

    # Move the feature_names into the fit method!
    model.fit(X_fit, t=t, feature_names=feature_names)

    print("\n--- Learned PDE Model (SINDy) ---")
    model.print()
    print(f"\nSINDy Model R^2 Score: {model.score(X_fit, t=t):.4f}")

    return model
