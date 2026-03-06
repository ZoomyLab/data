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

    Args:
        uniform_n_eta (int, optional): If provided, interpolates the extracted 1D slice
                                       onto a uniform dimensionless grid [0, 1] of size N.
                                       This neutralizes moving mesh / changing point counts.

    Returns:
        1D: (sweep_pos [N], values [N], lengths [N]) -> All 1D NumPy arrays.
        2D: (centers [List of [N,2]], values [List of [N]], lengths [List of [N]])
            -> Each list contains one NumPy array per sweep station.
    """
    if self.n_cells == 0:
        raise ValueError("The mesh has 0 cells.")

    # 1. Native Field Detection (No PyVista Smearing!)
    if field_name in self.point_data and field_name in self.cell_data:
        is_point = use_point_data  # Tie-breaker if it exists in both
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
        # Get the native coordinates
        pos = self.points if is_point else self.cell_centers().points

        # Determine sweep coordinate
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

        # Arc-length / Integration weights
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

        # --- NEW: Uniform Unit Interval Mapping (eta) ---
        if uniform_n_eta is not None and len(sorted_sweep) > 1:
            sweep_min, sweep_max = sorted_sweep[0], sorted_sweep[-1]
            # Dimensionless raw grid
            eta_raw = (sorted_sweep - sweep_min) / (sweep_max - sweep_min + 1e-12)
            eta_grid = np.linspace(0, 1, uniform_n_eta)

            # Interpolate values natively
            if sorted_vals.ndim > 1:  # Vector field
                interp_vals = np.zeros((uniform_n_eta, sorted_vals.shape[1]))
                for dim in range(sorted_vals.shape[1]):
                    interp_vals[:, dim] = np.interp(
                        eta_grid, eta_raw, sorted_vals[:, dim]
                    )
            else:  # Scalar field
                interp_vals = np.interp(eta_grid, eta_raw, sorted_vals)

            sorted_sweep = eta_grid
            sorted_vals = interp_vals

            # Recompute integration weights for uniform grid
            lens = np.full(uniform_n_eta, 1.0 / (uniform_n_eta - 1))
            lens[0] /= 2.0
            lens[-1] /= 2.0

            # Interpolate the physical arc_dist to match the new grid size
            arc_dist = np.interp(eta_grid, eta_raw, arc_dist)

        # Logic for returning coordinates
        if normal is None:
            return sorted_sweep, sorted_vals, lens
        else:
            # When recursing from 2D, return the [Sweep, Arc] pair
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

        # Project all cell centers onto the sweep vector to group slices
        centers = self.cell_centers().points
        all_projs = np.dot(centers - origin_pt, n_vec)

        # Round to 5 decimals to group cells that belong to the same column
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
    """
    Integrates a 2D mesh along a sweep line.
    Powered natively by NumPy arrays extracted via .to_np() for unbreakable stability.
    """
    if self.n_cells == 0:
        raise ValueError("Cannot integrate an empty mesh.")

    # Parse normal for 3D point reconstruction later
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

    # Compute integrals for every field using to_np
    for i, field in enumerate(fields_to_integrate):
        try:
            # 🚀 Extract all local coordinates, values, and lengths cleanly
            centers, values, lengths = self.to_np(field, normal=normal, origin=origin)
        except ValueError:
            continue  # Skip fields that fail slicing

        # Extract the sweep coordinates just once to rebuild the 3D line
        if valid_sweep_projs is None:
            # centers[j][0, 0] accesses the sweep_dist for the j-th slice
            valid_sweep_projs = [c[0, 0] for c in centers]

        field_integrals = []
        # Pure NumPy Riemann Sums!
        for v, dx in zip(values, lengths):
            if v.ndim > 1:  # Handle 3D vector fields cleanly
                field_integrals.append(np.sum(v * dx[:, np.newaxis], axis=0))
            else:  # Handle 1D scalar fields
                field_integrals.append(np.sum(v * dx))

        integrated_values[field] = np.stack(field_integrals)

    if valid_sweep_projs is None or len(valid_sweep_projs) == 0:
        raise RuntimeError("Integration failed. No valid slices were generated.")

    # Reconstruct the 3D points exactly along the sweep line
    valid_points = np.array([origin_pt + proj * norm_vec for proj in valid_sweep_projs])
    n_points = len(valid_points)

    # Build connectivity and the final 1D PyVista Mesh
    lines = np.empty((n_points - 1, 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(n_points - 1)
    lines[:, 2] = np.arange(1, n_points)

    line_mesh = pv.PolyData(valid_points, lines=lines.flatten())

    for field, vals in integrated_values.items():
        line_mesh.point_data[field] = vals

    # Clear scalar flag to protect future filters
    line_mesh.point_data.active_scalars_name = None
    return line_mesh


# Attach it natively to PyVista alongside your other custom methods
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

        # Load the exact physical times from the OpenFOAM folders
        self.of_times = self._get_openfoam_times()

        print(f"Initialized OFVTK: Found {self.num_timesteps} files.")
        if len(self.of_times) != self.num_timesteps:
            print(
                f"Warning: Found {len(self.of_times)} OpenFOAM time folders but {self.num_timesteps} VTK files. Time mapping may fallback to filenames."
            )

    def _get_openfoam_times(self):
        """
        Scans the parent directory (OpenFOAM root) for time directories
        (e.g., '0', '0.1', '36.2') and sorts them numerically.
        """
        # The parent of the VTK folder is usually the OpenFOAM root directory
        of_root = os.path.dirname(os.path.abspath(self.folder_path))

        time_folders = []
        import re

        for item in os.listdir(of_root):
            item_path = os.path.join(of_root, item)
            # Check if it's a directory and its name is strictly a number
            if os.path.isdir(item_path) and re.match(r"^[-+]?(?:\d*\.\d+|\d+)$", item):
                time_folders.append(float(item))

        # Sort them numerically so they perfectly align with the VTK files
        time_folders.sort()
        return time_folders

    def _natural_sort_key(self, s):
        """Helper to sort files numerically rather than purely alphabetically."""
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    def _get_sorted_files(self):
        """Collects and sorts the files."""
        search_pattern = os.path.join(self.folder_path, f"*{self.extension}")
        files = glob.glob(search_pattern)
        return sorted(files, key=self._natural_sort_key)

    def _vector_field_to_scalar_field(self, pv_obj, field_name):
        """
        Splits a 3D vector field into three scalar fields (_x, _y, _z)
        safely mapping to wherever the field is natively stored.
        """
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
        """
        Returns the physical simulation time for the given timestep index.
        """
        if index < -self.num_timesteps or index >= self.num_timesteps:
            raise IndexError(
                f"Index {index} out of bounds for {self.num_timesteps} timesteps."
            )

        # Handle negative indexing for array lookup
        actual_index = index if index >= 0 else self.num_timesteps + index

        # 1. Primary Method: 1-to-1 mapping with the OpenFOAM root time directories
        if len(self.of_times) == self.num_timesteps:
            return self.of_times[actual_index]

        # 2. Fallback: If folder counts mismatch, fallback to regexing the filename
        file_path = self.file_paths[index]
        basename = os.path.basename(file_path)
        import re

        matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", basename)

        if matches:
            return float(matches[-1])

        raise ValueError(f"Could not determine time for {basename}")

    def get_time_step(self, index):
        """
        Loads the PyVista object for the given timestep index (supports negative indexing)
        and automatically unpacks all vector fields into separate scalar components.
        """
        if index < -self.num_timesteps or index >= self.num_timesteps:
            raise IndexError(
                f"Index {index} out of bounds for {self.num_timesteps} timesteps."
            )

        file_to_load = self.file_paths[index]
        mesh = pv.read(file_to_load)

        # Unpack vector fields into scalars natively
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
    """
    Returns a list of available cell data field names for a given PyVista object.
    """
    return list((list(pv_obj.cell_data.keys()), list(pv_obj.point_data.keys())))


def plot(ax, pv_obj, field_name):
    # (Plot function remains identical and safe!)
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
    # (Remains identical)
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


def extract_unified_3d_dataset(sim, times, n_eta=50, save_path="unified_3d_data.npz"):
    print(f"\nBuilding UNIFIED 3D Dataset via Native PyVista...")
    all_nut, all_md, all_U, all_dU_dy, all_dU_dx, all_z, all_h, all_ustar = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    eta_grid = np.linspace(0, 1, n_eta)

    for s in times:
        t_current = sim.get_time(s)
        s_prev = s - 1 if s > 0 else -1
        dt = (
            t_current - sim.get_time(s_prev)
            if s > 0
            else sim.get_time(1) - sim.get_time(0)
        )

        d_3d = sim.get_time_step(s)
        d_3d_prev = sim.get_time_step(s_prev)

        # 1. Bed Map
        wss_pts = d_3d.point_data.get("wallShearStress", None)
        if wss_pts is None:
            continue
        wss_mag = np.linalg.norm(wss_pts, axis=1)
        y_min_mesh = np.min(d_3d.points[:, 1])
        bed_mask = (wss_mag > 1e-8) & (d_3d.points[:, 1] < y_min_mesh + 0.05)
        if not np.any(bed_mask):
            continue
        bed_x_coords, bed_ustar_values = (
            d_3d.points[bed_mask, 0],
            np.sqrt(wss_mag[bed_mask]),
        )

        # 2. Compute TRUE OpenFOAM gradients natively
        d_3d = d_3d.compute_derivative(scalars="nut", gradient="grad_nut")
        d_3d = d_3d.compute_derivative(scalars="U", gradient="grad_U")
        target = d_3d.point_data if "grad_nut" in d_3d.point_data else d_3d.cell_data
        d_3d.point_data["advection_term"] = np.sum(
            target["U"] * target["grad_nut"], axis=1
        )

        if (
            "alpha.water" in get_available_fields(d_3d)[0]
            or "alpha.water" in get_available_fields(d_3d)[1]
        ):
            d_3d = d_3d.threshold(0.5, scalars="alpha.water")
            d_3d_prev = d_3d_prev.threshold(0.5, scalars="alpha.water")

        x_stations = np.linspace(
            np.min(bed_x_coords) + 0.1, np.max(bed_x_coords) - 0.1, 40
        )

        for x in x_stations:
            slc = d_3d.slice(normal="x", origin=(x, 0, 0.05))
            if slc.n_points == 0:
                continue
            y_bed, y_surf = np.min(slc.points[:, 1]), np.max(slc.points[:, 1])
            h = y_surf - y_bed
            if h < 0.05:
                continue

            line_curr = d_3d.sample_over_line(
                (x, y_bed, 0.05), (x, y_surf, 0.05), resolution=n_eta - 1
            )
            line_prev = d_3d_prev.sample_over_line(
                (x, y_bed, 0.05), (x, y_surf, 0.05), resolution=n_eta - 1
            )

            nut_curr = line_curr.point_data["nut"]
            nut_prev = line_prev.point_data["nut"]
            adv = line_curr.point_data.get("advection_term", np.zeros_like(nut_curr))
            U_x = line_curr.point_data["U"][:, 0]

            # grad_U is a 9-component tensor [xx, xy, xz, yx, yy, yz, zx, zy, zz]
            # Component 0 is dUx/dx, Component 1 is dUx/dy
            grad_U = line_curr.point_data.get("grad_U", np.zeros((len(nut_curr), 9)))
            dU_dx = grad_U[:, 0]
            dU_dy = grad_U[:, 1]

            md = (nut_curr - nut_prev) / dt + adv
            u_star_local = np.interp(x, bed_x_coords, bed_ustar_values)

            all_nut.append(nut_curr)
            all_md.append(md)
            all_U.append(U_x)
            all_dU_dy.append(dU_dy)
            all_dU_dx.append(dU_dx)
            all_z.append(eta_grid)
            all_h.append(h)
            all_ustar.append(u_star_local)

    data_dict = {
        "nut": np.array(all_nut),
        "md": np.array(all_md),
        "U": np.array(all_U),
        "dU_dy": np.array(all_dU_dy),
        "dU_dx": np.array(all_dU_dx),
        "z": np.array(all_z),
        "h": np.array(all_h),
        "ustar": np.array(all_ustar),
    }
    np.savez(save_path, **data_dict)
    return data_dict


def load_or_build_unified(sim, times, n_eta=50):
    path = "data.npz"
    if not os.path.exists(path):
        extract_unified_3d_dataset(sim, times, n_eta, path)
    data = np.load(path)
    return {k: data[k] for k in data.files}
