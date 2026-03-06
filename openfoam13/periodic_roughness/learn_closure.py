from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np


def train_3d_pde_operator(data_path="unified_3d_data.npz", alpha=1e-3):
    data = np.load(data_path)
    nut = data["nut"].flatten()
    md = data["md"].flatten()
    U = data["U"].flatten()
    dU_dy = data["dU_dy"].flatten()
    dU_dx = data["dU_dx"].flatten()
    h = np.repeat(data["h"], data["nut"].shape[1])

    inv_h = 1.0 / h

    features_3d = {
        "nut * |dU_dy|": nut * np.abs(dU_dy),
        "nut * |dU_dx|": nut * np.abs(dU_dx),
        "nut * dU_dx": nut * dU_dx,
        "nut^2 / h^2": nut**2 * inv_h**2,
        "|U| * nut / h": np.abs(U) * nut * inv_h,
        "U * nut / h": U * nut * inv_h,
        "U^2": U**2,
    }

    X_3d = np.column_stack(list(features_3d.values()))
    Y_3d = md

    # Filter valid
    valid_mask = np.all(np.isfinite(X_3d), axis=1) & np.isfinite(Y_3d)
    X_clean, Y_clean = X_3d[valid_mask], Y_3d[valid_mask]

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_clean)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_clean.reshape(-1, 1)).flatten()

    print(f"\n{'=' * 60}\nDISCOVERING DIMENSIONALLY PURE 3D SOURCE TERMS\n{'=' * 60}")
    selector = Lasso(alpha=alpha, fit_intercept=False, max_iter=200000)
    selector.fit(X_scaled, Y_scaled)

    mask = np.abs(selector.coef_) > 1e-5
    if not np.any(mask):
        print("No terms selected. Try decreasing alpha.")
        return None, None, None

    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_clean[:, mask], Y_clean)

    print("D(nut)/Dt = ")
    feature_names = list(features_3d.keys())
    surviving_names = [feature_names[i] for i in range(len(mask)) if mask[i]]
    for name, coef in zip(surviving_names, ols.coef_):
        print(f"    {coef:+.5e} * [{name}]")

    print(f"\n3D Source Term R^2: {ols.score(X_clean[:, mask], Y_clean):.4f}")
    return ols, surviving_names, ols.coef_
