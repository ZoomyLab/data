import pyvista as pv
import xml.etree.ElementTree as ET
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def _pv_to_np(
    self,
    field_name,
    normal=None,
    origin=(0, 0, 0),
    use_point_data=False,
    uniform_n_eta=None,
):
    """
    Extracts geometric and field data into NumPy arrays.
    """
    if self.n_cells == 0:
        raise ValueError("The mesh has 0 cells.")

    # 1. Native Field Detection
    if field_name in self.point_data and field_name in self.cell_data:
        is_point = use_point_data
    elif field_name in self.point_data:
        is_point = True
    elif field_name in self.cell_data:
        is_point = False
    else:
        raise KeyError(f"Field '{field_name}' not found in points or cells.")

    vals = self.point_data[field_name] if is_point else self.cell_data[field_name]
    max_dim = max([self.get_cell(i).dimension for i in range(min(10, self.n_cells))])

    # ==========================================
    # 1D Object Logic (The Base Case)
    # ==========================================
    if max_dim == 1:
        pos = self.points if is_point else self.cell_centers().points

        if normal is not None:
            if isinstance(normal, str):
                axis_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
                n_vec = np.array(axis_map[normal.lower()])
            else:
                n_vec = np.array(normal) / np.linalg.norm(normal)
            sweep_coords = np.dot(pos - np.array(origin), n_vec)
        else:
            dom_axis = np.argmax(np.ptp(pos, axis=0))
            sweep_coords = pos[:, dom_axis]

        sort_idx = np.argsort(sweep_coords)
        sorted_sweep = sweep_coords[sort_idx]
        sorted_vals = vals[sort_idx]

        if len(pos) > 1:
            steps = np.linalg.norm(np.diff(pos[sort_idx], axis=0), axis=1)
            arc_dist = np.concatenate(([0], np.cumsum(steps)))
            if is_point:
                lens = np.zeros_like(arc_dist)
                lens[0], lens[-1] = steps[0] / 2, steps[-1] / 2
                if len(arc_dist) > 2:
                    lens[1:-1] = (steps[:-1] + steps[1:]) / 2
            else:
                lens = self.compute_cell_sizes(length=True).cell_data["Length"][
                    sort_idx
                ]
        else:
            arc_dist, lens = np.array([0.0]), np.array([0.0])

        if uniform_n_eta is not None and len(sorted_sweep) > 1:
            sweep_min, sweep_max = sorted_sweep[0], sorted_sweep[-1]
            eta_raw = (sorted_sweep - sweep_min) / (sweep_max - sweep_min + 1e-12)
            eta_grid = np.linspace(0, 1, uniform_n_eta)

            if sorted_vals.ndim > 1:
                interp_vals = np.zeros((uniform_n_eta, sorted_vals.shape[1]))
                for dim in range(sorted_vals.shape[1]):
                    interp_vals[:, dim] = np.interp(
                        eta_grid, eta_raw, sorted_vals[:, dim]
                    )
            else:
                interp_vals = np.interp(eta_grid, eta_raw, sorted_vals)

            sorted_sweep = eta_grid
            sorted_vals = interp_vals

            lens = np.full(uniform_n_eta, 1.0 / (uniform_n_eta - 1))
            lens[0] /= 2.0
            lens[-1] /= 2.0

            arc_dist = np.interp(eta_grid, eta_raw, arc_dist)

        if normal is None:
            return sorted_sweep, sorted_vals, lens
        else:
            return np.column_stack((sorted_sweep, arc_dist)), sorted_vals, lens

    # ==========================================
    # 2D Object Logic (The Sweep Case)
    # ==========================================
    elif max_dim == 2:
        if normal is None:
            raise ValueError("Normal required for 2D sweep.")

        if isinstance(normal, str):
            axis_map = {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            }
            n_vec = np.array(axis_map[normal.lower()])
        else:
            n_vec = np.array(normal) / np.linalg.norm(normal)

        origin_pt = np.array(origin)
        centers = self.cell_centers().points
        all_projs = np.dot(centers - origin_pt, n_vec)

        unique_projs = np.unique(np.round(all_projs, decimals=5))

        c_list, v_list, l_list = [], [], []

        for proj in unique_projs:
            station_pt = origin_pt + proj * n_vec
            slc = self.slice(normal=n_vec, origin=station_pt)

            if slc.n_cells > 0:
                c, v, l = slc.to_np(
                    field_name,
                    normal=normal,
                    origin=origin,
                    use_point_data=use_point_data,
                    uniform_n_eta=uniform_n_eta,
                )
                c_list.append(c)
                v_list.append(v)
                l_list.append(l)

        return c_list, v_list, l_list
    else:
        raise ValueError(f".to_np() supports 1D/2D, found {max_dim}.")


def _pv_integrate_mesh(self, normal, origin=(0, 0, 0)):
    if self.n_cells == 0:
        raise ValueError("Cannot integrate an empty mesh.")

    if isinstance(normal, str):
        axis_map = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
        norm_vec = np.array(axis_map[normal.lower()])
    else:
        norm_vec = np.array(normal, dtype=float)
        norm_vec /= np.linalg.norm(norm_vec)

    origin_pt = np.array(origin, dtype=float)

    fields_to_integrate = list(self.cell_data.keys())
    integrated_values = {}
    valid_sweep_projs = None

    for i, field in enumerate(fields_to_integrate):
        try:
            centers, values, lengths = self.to_np(field, normal=normal, origin=origin)
        except ValueError:
            continue

        if valid_sweep_projs is None:
            valid_sweep_projs = [c[0, 0] for c in centers]

        field_integrals = []
        for v, dx in zip(values, lengths):
            if v.ndim > 1:
                field_integrals.append(np.sum(v * dx[:, np.newaxis], axis=0))
            else:
                field_integrals.append(np.sum(v * dx))

        integrated_values[field] = np.stack(field_integrals)

    if valid_sweep_projs is None or len(valid_sweep_projs) == 0:
        raise RuntimeError("Integration failed. No valid slices were generated.")

    valid_points = np.array([origin_pt + proj * norm_vec for proj in valid_sweep_projs])
    n_points = len(valid_points)

    lines = np.empty((n_points - 1, 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(n_points - 1)
    lines[:, 2] = np.arange(1, n_points)

    line_mesh = pv.PolyData(valid_points, lines=lines.flatten())

    for field, vals in integrated_values.items():
        line_mesh.point_data[field] = vals

    line_mesh.point_data.active_scalars_name = None
    return line_mesh


# Attach it natively to PyVista
pv.DataSet.to_np = _pv_to_np
pv.DataSet.integrate = _pv_integrate_mesh


class VTKOF:
    def __init__(self, folder_path, extension=".vtk"):
        self.folder_path = folder_path
        self.extension = extension
        self.file_paths = self._get_sorted_files()
        self.num_timesteps = len(self.file_paths)

        if self.num_timesteps == 0:
            raise FileNotFoundError(
                f"No files ending in '{extension}' found in {folder_path}."
            )

        self.of_times = self._get_openfoam_times()

        print(f"Initialized OFVTK: Found {self.num_timesteps} files.")
        if len(self.of_times) != self.num_timesteps:
            print(
                f"Warning: Found {len(self.of_times)} OpenFOAM time folders but {self.num_timesteps} VTK files. Time mapping may fallback to filenames."
            )

    def _get_openfoam_times(self):
        of_root = os.path.dirname(os.path.abspath(self.folder_path))
        time_folders = []
        for item in os.listdir(of_root):
            item_path = os.path.join(of_root, item)
            if os.path.isdir(item_path) and re.match(r"^[-+]?(?:\d*\.\d+|\d+)$", item):
                time_folders.append(float(item))
        time_folders.sort()
        return time_folders

    def _natural_sort_key(self, s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    def _get_sorted_files(self):
        search_pattern = os.path.join(self.folder_path, f"*{self.extension}")
        files = glob.glob(search_pattern)
        return sorted(files, key=self._natural_sort_key)

    def _vector_field_to_scalar_field(self, pv_obj, field_name):
        if field_name in pv_obj.cell_data:
            vector_data = pv_obj.cell_data[field_name]
            pv_obj.cell_data[f"{field_name}_x"] = vector_data[:, 0]
            pv_obj.cell_data[f"{field_name}_y"] = vector_data[:, 1]
            pv_obj.cell_data[f"{field_name}_z"] = vector_data[:, 2]

        if field_name in pv_obj.point_data:
            vector_data = pv_obj.point_data[field_name]
            pv_obj.point_data[f"{field_name}_x"] = vector_data[:, 0]
            pv_obj.point_data[f"{field_name}_y"] = vector_data[:, 1]
            pv_obj.point_data[f"{field_name}_z"] = vector_data[:, 2]

    def size(self):
        return self.num_timesteps

    def get_time(self, index):
        if index < -self.num_timesteps or index >= self.num_timesteps:
            raise IndexError(
                f"Index {index} out of bounds for {self.num_timesteps} timesteps."
            )
        actual_index = index if index >= 0 else self.num_timesteps + index
        if len(self.of_times) == self.num_timesteps:
            return self.of_times[actual_index]

        file_path = self.file_paths[index]
        basename = os.path.basename(file_path)
        matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", basename)
        if matches:
            return float(matches[-1])
        raise ValueError(f"Could not determine time for {basename}")

    def get_time_step(self, index):
        if index < -self.num_timesteps or index >= self.num_timesteps:
            raise IndexError(
                f"Index {index} out of bounds for {self.num_timesteps} timesteps."
            )
        file_to_load = self.file_paths[index]
        mesh = pv.read(file_to_load)

        for field_name in list(mesh.cell_data.keys()):
            arr = mesh.cell_data[field_name]
            if len(arr.shape) == 2 and arr.shape[1] == 3:
                self._vector_field_to_scalar_field(mesh, field_name)

        for field_name in list(mesh.point_data.keys()):
            arr = mesh.point_data[field_name]
            if len(arr.shape) == 2 and arr.shape[1] == 3:
                self._vector_field_to_scalar_field(mesh, field_name)

        return mesh


def get_available_fields(pv_obj):
    return list((list(pv_obj.cell_data.keys()), list(pv_obj.point_data.keys())))


def plot(ax, pv_obj, field_name):
    if field_name not in pv_obj.cell_data:
        raise ValueError(
            f"Field '{field_name}' not found in the object's cell_data. "
            f"Available fields: {list(pv_obj.cell_data.keys())}"
        )

    sample_size = min(10, pv_obj.n_cells)
    cell_dims = [pv_obj.get_cell(i).dimension for i in range(sample_size)]
    max_dim = max(cell_dims)

    cell_values = pv_obj.cell_data[field_name]

    if max_dim == 1:
        centers = pv_obj.cell_centers().points
        spreads = np.ptp(centers, axis=0)
        dominant_axis = np.argmax(spreads)
        sort_indices = np.argsort(centers[:, dominant_axis])

        sorted_centers = centers[sort_indices]
        distances = np.linalg.norm(sorted_centers - sorted_centers[0], axis=1)
        sorted_values = cell_values[sort_indices]

        ax.plot(distances, sorted_values, marker="o", linestyle="-", markersize=4)
        ax.set_xlabel("Distance along line")
        ax.set_ylabel(field_name)
        ax.set_title(f"1D Plot: {field_name}")
        ax.grid(True)

    elif max_dim == 2:
        points = pv_obj.points
        spreads = np.ptp(points, axis=0)
        planar_axes = np.argsort(spreads)[1:]
        pts_2d = points[:, planar_axes]

        verts = []
        for i in range(pv_obj.n_cells):
            cell = pv_obj.get_cell(i)
            verts.append(pts_2d[cell.point_ids])

        collection = PolyCollection(
            verts,
            array=cell_values,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.2,
        )

        ax.add_collection(collection)
        ax.autoscale_view()
        cbar = plt.colorbar(collection, ax=ax)
        cbar.set_label(field_name)

        axis_names = ["X", "Y", "Z"]
        ax.set_xlabel(f"Coordinate {axis_names[planar_axes[0]]}")
        ax.set_ylabel(f"Coordinate {axis_names[planar_axes[1]]}")
        ax.set_title(f"2D Mesh Slice: {field_name}")
        ax.set_aspect("equal", "box")
    else:
        raise ValueError(
            f"Unsupported cell dimension {max_dim}. Expected 1D or 2D slice objects."
        )


def transpose_plot(ax):
    for line in ax.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        line.set_xdata(y_data)
        line.set_ydata(x_data)

    for collection in ax.collections:
        paths = collection.get_paths()
        new_verts = []
        for path in paths:
            swapped = path.vertices[:, [1, 0]]
            new_verts.append(swapped)
        collection.set_verts(new_verts)

    old_xlabel = ax.get_xlabel()
    old_ylabel = ax.get_ylabel()
    ax.set_xlabel(old_ylabel)
    ax.set_ylabel(old_xlabel)

    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()
    ax.set_xlim(old_ylim)
    ax.set_ylim(old_xlim)
    ax.figure.canvas.draw_idle()


def load_or_build_data(sim, times, n_stations=40, n_eta=50):
    path = "data.npz"
    if not os.path.exists(path):
        extract_3d_dataset(
            sim, times, n_eta=n_eta, n_stations=n_stations, save_path=path
        )
    data = np.load(path)
    return {k: data[k] for k in data.files}


def extract_3d_dataset(sim, times, n_eta=50, n_stations=40, save_path="data.npz"):
    print(f"\nBuilding 3D Dataset [Time x Space x Z] via Native PyVista...")
    n_times = len(times)

    # Pre-allocate strict 3D arrays
    all_nut = np.zeros((n_times, n_stations, n_eta))
    all_md = np.zeros((n_times, n_stations, n_eta))
    all_U = np.zeros((n_times, n_stations, n_eta))

    all_k = np.zeros((n_times, n_stations, n_eta))
    all_md_k = np.zeros((n_times, n_stations, n_eta))

    all_omega = np.zeros((n_times, n_stations, n_eta))
    all_md_omega = np.zeros((n_times, n_stations, n_eta))

    # Pre-allocate Gradients
    all_dU_dx, all_dU_dy = np.zeros_like(all_nut), np.zeros_like(all_nut)
    all_d2U_dx2, all_d2U_dy2 = np.zeros_like(all_nut), np.zeros_like(all_nut)

    all_dnut_dx, all_dnut_dy = np.zeros_like(all_nut), np.zeros_like(all_nut)
    all_d2nut_dx2, all_d2nut_dy2 = np.zeros_like(all_nut), np.zeros_like(all_nut)

    all_dk_dx, all_dk_dy = np.zeros_like(all_nut), np.zeros_like(all_nut)
    all_d2k_dx2, all_d2k_dy2 = np.zeros_like(all_nut), np.zeros_like(all_nut)

    all_domega_dx, all_domega_dy = np.zeros_like(all_nut), np.zeros_like(all_nut)
    all_d2omega_dx2, all_d2omega_dy2 = np.zeros_like(all_nut), np.zeros_like(all_nut)

    all_h = np.zeros((n_times, n_stations))
    all_ustar = np.zeros((n_times, n_stations))

    eta_grid = np.linspace(0, 1, n_eta)

    for t_idx, s in enumerate(times):
        t_current = sim.get_time(s)
        s_prev = s - 1 if s > 0 else -1
        dt = (
            t_current - sim.get_time(s_prev)
            if s > 0
            else sim.get_time(1) - sim.get_time(0)
        )

        print(f"  Processing time {t_current:.2f} [{t_idx + 1}/{n_times}]...")
        d_3d = sim.get_time_step(s)
        d_3d_prev = sim.get_time_step(s_prev)

        # Bed Map
        wss_pts = d_3d.point_data.get("wallShearStress", None)
        wss_mag = np.linalg.norm(wss_pts, axis=1)
        y_min_mesh = np.min(d_3d.points[:, 1])
        bed_mask = (wss_mag > 1e-8) & (d_3d.points[:, 1] < y_min_mesh + 0.05)
        bed_x_coords, bed_ustar_values = (
            d_3d.points[bed_mask, 0],
            np.sqrt(wss_mag[bed_mask]),
        )

        # Compute Gradients Natively
        d_3d = d_3d.compute_derivative(scalars="nut", gradient="grad_nut")
        d_3d = d_3d.compute_derivative(scalars="grad_nut", gradient="grad2_nut")

        if "k" in d_3d.point_data or "k" in d_3d.cell_data:
            d_3d = d_3d.compute_derivative(scalars="k", gradient="grad_k")
            d_3d = d_3d.compute_derivative(scalars="grad_k", gradient="grad2_k")

        if "omega" in d_3d.point_data or "omega" in d_3d.cell_data:
            d_3d = d_3d.compute_derivative(scalars="omega", gradient="grad_omega")
            d_3d = d_3d.compute_derivative(scalars="grad_omega", gradient="grad2_omega")

        if "U_x" not in d_3d.point_data and "U_x" not in d_3d.cell_data:
            target_data = d_3d.point_data if "U" in d_3d.point_data else d_3d.cell_data
            target_data["U_x"] = target_data["U"][:, 0]
        d_3d = d_3d.compute_derivative(scalars="U_x", gradient="grad_Ux")
        d_3d = d_3d.compute_derivative(scalars="grad_Ux", gradient="grad2_Ux")

        # Advection Terms
        target = d_3d.point_data if "grad_nut" in d_3d.point_data else d_3d.cell_data
        d_3d.point_data["adv_nut"] = np.sum(target["U"] * target["grad_nut"], axis=1)
        if "grad_k" in target:
            d_3d.point_data["adv_k"] = np.sum(target["U"] * target["grad_k"], axis=1)
        if "grad_omega" in target:
            d_3d.point_data["adv_omega"] = np.sum(
                target["U"] * target["grad_omega"], axis=1
            )

        x_stations = np.linspace(
            np.min(bed_x_coords) + 0.1, np.max(bed_x_coords) - 0.1, n_stations
        )

        for x_idx, x in enumerate(x_stations):
            slc = d_3d.slice(normal="x", origin=(x, 0, 0.05))
            y_bed, y_surf = np.min(slc.points[:, 1]), np.max(slc.points[:, 1])
            h = y_surf - y_bed

            line_curr = d_3d.sample_over_line(
                (x, y_bed, 0.05), (x, y_surf, 0.05), resolution=n_eta - 1
            )
            line_prev = d_3d_prev.sample_over_line(
                (x, y_bed, 0.05), (x, y_surf, 0.05), resolution=n_eta - 1
            )

            pd_curr, pd_prev = line_curr.point_data, line_prev.point_data

            # Base variables
            all_nut[t_idx, x_idx, :] = pd_curr["nut"]
            all_U[t_idx, x_idx, :] = pd_curr["U"][:, 0]
            all_k[t_idx, x_idx, :] = pd_curr.get("k", np.zeros_like(eta_grid))
            all_omega[t_idx, x_idx, :] = pd_curr.get("omega", np.zeros_like(eta_grid))

            # Material Derivatives
            all_md[t_idx, x_idx, :] = (
                pd_curr["nut"] - pd_prev["nut"]
            ) / dt + pd_curr.get("adv_nut", 0)
            all_md_k[t_idx, x_idx, :] = (
                pd_curr.get("k", 0) - pd_prev.get("k", 0)
            ) / dt + pd_curr.get("adv_k", 0)
            all_md_omega[t_idx, x_idx, :] = (
                pd_curr.get("omega", 0) - pd_prev.get("omega", 0)
            ) / dt + pd_curr.get("adv_omega", 0)

            # Extract Gradients
            for prefix, name in zip(
                ["U", "nut", "k", "omega"], ["Ux", "nut", "k", "omega"]
            ):
                if f"grad_{name}" in pd_curr:
                    locals()[f"all_d{prefix}_dx"][t_idx, x_idx, :] = pd_curr[
                        f"grad_{name}"
                    ][:, 0]
                    locals()[f"all_d{prefix}_dy"][t_idx, x_idx, :] = pd_curr[
                        f"grad_{name}"
                    ][:, 1]
                    locals()[f"all_d2{prefix}_dx2"][t_idx, x_idx, :] = pd_curr[
                        f"grad2_{name}"
                    ][:, 0]  # xx
                    locals()[f"all_d2{prefix}_dy2"][t_idx, x_idx, :] = pd_curr[
                        f"grad2_{name}"
                    ][:, 4]  # yy

            all_h[t_idx, x_idx] = h
            all_ustar[t_idx, x_idx] = np.interp(x, bed_x_coords, bed_ustar_values)

    all_dh_dx = np.gradient(all_h, x_stations, axis=1)

    data_dict = {
        "nut": all_nut,
        "md": all_md,
        "U": all_U,
        "k": all_k,
        "md_k": all_md_k,
        "omega": all_omega,
        "md_omega": all_md_omega,
        "dU_dx": all_dU_dx,
        "dU_dy": all_dU_dy,
        "d2U_dx2": all_d2U_dx2,
        "d2U_dy2": all_d2U_dy2,
        "dnut_dx": all_dnut_dx,
        "dnut_dy": all_dnut_dy,
        "d2nut_dx2": all_d2nut_dx2,
        "d2nut_dy2": all_d2nut_dy2,
        "dk_dx": all_dk_dx,
        "dk_dy": all_dk_dy,
        "d2k_dx2": all_d2k_dx2,
        "d2k_dy2": all_d2k_dy2,
        "domega_dx": all_domega_dx,
        "domega_dy": all_domega_dy,
        "d2omega_dx2": all_d2omega_dx2,
        "d2omega_dy2": all_d2omega_dy2,
        "dh_dx": all_dh_dx,
        "z": eta_grid,
        "h": all_h,
        "ustar": all_ustar,
    }
    np.savez(save_path, **data_dict)
    return data_dict
