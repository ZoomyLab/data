from vtk_core import *
from learn_closure import *
from evaluate_classical_models import *
# from algebraic_model import build_training_dataset, train_sparse_closure
# from pde_model import build_training_dataset_sindy, train_sindy_pde
# from pde_closure_material_derivative import (
#    build_material_dataset,
#    train_algebraic_pde,
#    evaluate_3d_reconstruction,
#    train_3d_pde_operator,
# )
# from evaluate_classical_models import (
#    extract_raw_3d_dataset,
#    evaluate_classical_models,
#    load_or_build_dataset,
# )

if __name__ == "__main__":
    sim = VTKOF("./VTK")

    # d_3d = sim.get_time_step(-1)
    # print(get_available_fields(d_3d))
    # d_3d_w = d_3d.threshold(0.001, scalars="alpha.water")
    # d_2d_w = d_3d_w.slice(normal="z", origin=(0, 0, 0.05))
    # d_2d = d_3d.slice(normal="z", origin=(0, 0, 0.05))
    # d_1d_w = d_2d_w.slice(normal="x", origin=(5, 0, 0.05))
    # d_1d_int = d_2d.integrate(normal="x", origin=(0, 0, 0.05))
    # fig, ax = plt.subplots(2, 2, constrained_layout=True)
    # plot(ax[0, 0], d_2d, "alpha.water")
    # plot(ax[1, 0], d_2d_w, "U_x")
    # plot(ax[1, 1], d_1d_w, "U_x")
    # plot(ax[0, 1], d_1d_int.ptc(), "alpha.water")
    # transpose_plot(ax[0, 0])
    # transpose_plot(ax[1, 0])
    # transpose_plot(ax[1, 1])
    # fig.savefig("slice.svg")

    # times = [1, -1]
    # times = range(1, sim.size(), 10)
    # timeline = np.zeros(len(times) + 1)
    # beta = np.zeros(len(times) + 1)
    # beta[0] = 1.0

    # def compute_beta(d_2d_w):
    #    centers, ux, length = d_2d_w.to_np(
    #        "U_x", normal="x", origin=(0, 0, 0.05), use_point_data=True
    #    )
    #    _, alpha, _ = d_2d_w.to_np(
    #        "alpha.water", normal="x", origin=(0, 0, 0.05), use_point_data=True
    #    )
    #    idx = 2
    #    beta = 0
    #    # print(dist)
    #    h = np.sum(length[idx])
    #    # h = d_2d_w.integrate(normal="x", origin=(0, 0, 0.05)).point_data["alpha.water"][
    #    #    0
    #    # ]
    #    mean_u_sq = (1 / h * np.sum(alpha[idx] * ux[idx] * length[idx])) ** 2
    #    sq_u_mean = 1 / h * np.sum(alpha[idx] * ux[idx] * ux[idx] * length[idx])
    #    beta = sq_u_mean / mean_u_sq
    #    return beta

    # for i, s in enumerate(times):
    #    d_3d = sim.get_time_step(s)
    #    d_3d_w = d_3d.threshold(0.001, scalars="alpha.water")
    #    d_2d_w = d_3d_w.slice(normal="z", origin=(0, 0, 0.05))
    #    timeline[i + 1] = sim.get_time(s)
    #    beta[i + 1] = compute_beta(d_2d_w)

    # beta_final = beta[-1]
    # beta_init = beta[0]

    # def f_beta(t, tau):
    #    return beta_final + (beta_init - beta_final) * np.exp(-t / tau)

    # from scipy.optimize import curve_fit

    # popt, pcov = curve_fit(f_beta, timeline, beta)
    # print(popt, pcov)

    # fig, ax = plt.subplots()
    # ax.plot(timeline, beta, "*-")
    # ax.plot(timeline, f_beta(timeline, popt))
    # fig.savefig("beta.svg")

    times = range(1, sim.size(), 200)
    # ==========================================
    # Experiment 1: Algebraic / VAM Moment Model
    # ==========================================
    # print("\n--- Running Algebraic Model ---")
    # build_training_dataset(sim, times, save_path="closure_data.npy")
    # train_sparse_closure("closure_data.npy", alpha_penalty=1e-6)

    # ==========================================
    # Experiment 2: PDE / SINDy Model
    # ==========================================
    # print("\n--- Running PDE Discovery ---")
    ## build_training_dataset_sindy(sim, times, n_eta=50, save_path="sindy_pde_data.npz")
    # train_sindy_pde("sindy_pde_data.npz", threshold=1e-6)

    # 1. Build Data
    # build_material_dataset(
    #    sim, times, n_moments=3, save_path="material_derivative_data.npy"
    # )

    # models, masks, feat_names = train_algebraic_pde(
    #    "material_derivative_data.npy", n_moments=3, alpha=0.00001
    # )

    # evaluate_3d_reconstruction(
    #    "material_derivative_data.npy",
    #    trained_models=models,
    #    masks=masks,
    #    feature_names=feat_names,
    #    n_moments=3,
    # )

    # train_3d_pde_operator("material_derivative_data.npy", n_moments=3, alpha=0.001)

    # extract_raw_3d_dataset(sim, times, n_eta=50)
    # data = load_or_build_dataset(sim, times, n_eta=50)
    # evaluate_classical_models(data)

    data = load_or_build_unified(sim, times, n_eta=50)
    train_3d_pde_operator("data.npz", alpha=0.001)
    evaluate_classical_models(data)
    plot_comparison(data)
