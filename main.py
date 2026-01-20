import argparse
import os.path as osp

import numpy as np
import open3d as o3d
import pickle

from plyfile import PlyData
from skimage import color

from loguru import logger
import os

#################################
# work around for my wayland system
# need to force it to use x11 instead of wayland
os.environ["WAYLAND_DISPLAY"] = ""
os.environ["XDG_SESSION_TYPE"] = ""
# GLFW_PLATFORM = "x11"
#################################

np.random.seed(42)


def _read_ply(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = vertex.data.dtype.names
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
        np.float32
    )

    def _field(name_options):
        for name in name_options:
            if name in names:
                return vertex[name]
        return None

    r = _field(["red", "r"])
    g = _field(["green", "g"])
    b = _field(["blue", "b"])
    if r is None or g is None or b is None:
        return points, None

    colors = np.stack([r, g, b], axis=1).astype(np.float32)
    return points, colors


def _read_npy_or_npz(path):
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "points" in data:
            points = data["points"]
        else:
            key = list(data.keys())[0]
            points = data[key]
    else:
        points = data
    if points.shape[1] >= 6:
        return points[:, :3], points[:, 3:6]
    return points[:, :3], None


def load_point_cloud(path, point_limit=None):
    ext = osp.splitext(path)[1].lower()
    if ext == ".ply":
        points, colors = _read_ply(path)
    elif ext in {".npy", ".npz"}:
        points, colors = _read_npy_or_npz(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {path}")

    if point_limit is not None and points.shape[0] > point_limit:
        indices = np.random.permutation(points.shape[0])[:point_limit]
        points = points[indices]
        if colors is not None:
            colors = colors[indices]

    rgb = None
    if colors is None:
        hsv = np.zeros((points.shape[0], 3), dtype=np.float32)
    else:
        colors = colors.astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        rgb = colors
        hsv = color.rgb2hsv(colors).astype(np.float32)

    return points.astype(np.float32), hsv, rgb


def load_transform(path):
    if path is None:
        return np.eye(4, dtype=np.float32)
    ext = osp.splitext(path)[1].lower()
    if ext == ".npy":
        matrix = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if "transform" in data:
            matrix = data["transform"]
        else:
            key = list(data.keys())[0]
            matrix = data[key]
    else:
        matrix = np.loadtxt(path)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4, got {matrix.shape}")
    return matrix


def apply_transform_np(points, transform):
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    hom = np.concatenate([points, ones], axis=1)
    aligned = hom @ transform.T
    return aligned[:, :3]


def compute_corr_stats(ref_corr_points, src_corr_points, transform, acceptance_radius):
    if ref_corr_points.size == 0:
        return {
            "total_corr": 0,
            "inliers": 0,
            "outliers": 0,
            "inlier_ratio": 0.0,
        }
    aligned_src = apply_transform_np(src_corr_points, transform)
    dists = np.linalg.norm(ref_corr_points - aligned_src, axis=1)
    inliers = int(np.sum(dists < acceptance_radius))
    total = int(dists.shape[0])
    outliers = total - inliers
    inlier_ratio = float(inliers) / float(total) if total > 0 else 0.0
    return {
        "total_corr": total,
        "inliers": inliers,
        "outliers": outliers,
        "inlier_ratio": inlier_ratio,
    }


def visualize_pair(
    ref_points,
    src_points,
    est_transform,
    gt_transform,
    ref_corr,
    src_corr,
    max_corr,
    ref_rgb,
    src_rgb,
    use_raw_color,
):
    def make_pcd(points, color_rgb, per_point_rgb=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if per_point_rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(per_point_rgb.astype(np.float64))
        else:
            pcd.paint_uniform_color(color_rgb)
        return pcd

    def make_lineset(ref_pts, src_pts, color_rgb):
        if ref_pts.size == 0:
            return o3d.geometry.LineSet()
        n = ref_pts.shape[0]
        if max_corr is not None and n > max_corr:
            indices = np.random.permutation(n)[:max_corr]
            ref_pts = ref_pts[indices]
            src_pts = src_pts[indices]
        all_points = np.concatenate([ref_pts, src_pts], axis=0)
        lines = [[i, i + ref_pts.shape[0]] for i in range(ref_pts.shape[0])]
        colors = np.tile(np.asarray(color_rgb, dtype=np.float64), (len(lines), 1))
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    ref_color = np.asarray([9.0, 151.0, 247.0]) / 255.0
    src_color = np.asarray([221.0, 184.0, 34.0]) / 255.0
    est_color = np.asarray([245.0, 95.0, 95.0]) / 255.0

    use_raw = use_raw_color and ref_rgb is not None and src_rgb is not None
    ref_pcd = make_pcd(
        ref_points, ref_color, per_point_rgb=ref_rgb if use_raw else None
    )
    src_pcd = make_pcd(
        src_points, src_color, per_point_rgb=src_rgb if use_raw else None
    )
    est_src_pcd = make_pcd(
        apply_transform_np(src_points, est_transform),
        est_color,
        per_point_rgb=src_rgb if use_raw else None,
    )

    mode_geoms = {
        "raw": [ref_pcd, src_pcd],
        "est": [ref_pcd, est_src_pcd],
    }
    if gt_transform is not None:
        gt_src_pcd = make_pcd(
            apply_transform_np(src_points, gt_transform),
            est_color,
            per_point_rgb=src_rgb if use_raw else None,
        )
        mode_geoms["gt"] = [ref_pcd, gt_src_pcd]

    if ref_corr is not None and src_corr is not None:
        corr_lines = make_lineset(
            ref_corr, apply_transform_np(src_corr, est_transform), est_color
        )
        mode_geoms["corr"] = [ref_pcd, est_src_pcd, corr_lines]

    color_label = "raw-color" if use_raw else "distinct-color"
    help_text = f"Controls: 1=raw 2=est 3=gt 4=corr m/M=cycle q=close | {color_label}"
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"ColorPCR viewer | {help_text}")
    view_option = vis.get_view_control() or o3d.visualization.ViewControl()
    render_option = vis.get_render_option() or o3d.visualization.RenderOption()
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    render_option.point_size = 3.0

    mode_order = list(mode_geoms.keys())
    current_mode = {"value": mode_order[0]}

    def apply_mode(mode):
        current_mode["value"] = mode
        vis.clear_geometries()
        for geometry in mode_geoms[mode]:
            vis.add_geometry(geometry)
        view_option.set_front([0.0, -0.3, -1.0])
        view_option.set_up([0.0, 0.0, 1.0])
        view_option.set_zoom(0.4)
        vis.update_renderer()
        print(f"[Visualizer] Switched to {mode} mode. {help_text}")

    def cycle_mode(_):
        idx = mode_order.index(current_mode["value"])
        apply_mode(mode_order[(idx + 1) % len(mode_order)])
        return False

    for idx, mode in enumerate(mode_order):
        if idx < 9:
            vis.register_key_callback(
                ord(str(idx + 1)), lambda _, m=mode: apply_mode(m)
            )
    vis.register_key_callback(ord("m"), cycle_mode)
    vis.register_key_callback(ord("M"), cycle_mode)

    apply_mode(current_mode["value"])
    vis.run()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corr", default="corr.pkl", help="Correspondence file (.pkl)")
    parser.add_argument(
        "--ref",
        default="combined_pcd_1.ply",
        help="Reference point cloud (.ply/.npy/.npz)",
    )
    parser.add_argument(
        "--src",
        default="combined_pcd_3.ply",
        help="Source point cloud (.ply/.npy/.npz)",
    )
    parser.add_argument(
        "--transform", default=None, help="Optional GT transform file (4x4)"
    )
    parser.add_argument(
        "--point_limit", type=int, default=None, help="Max points per cloud"
    )
    parser.add_argument(
        "--save", default=None, help="Optional output .npz to save results"
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize in Open3D")
    parser.add_argument(
        "--vis_max_corr", type=int, default=2000, help="Max correspondences to draw"
    )
    parser.add_argument(
        "--vis_color_mode",
        choices=["raw", "distinct"],
        default="raw",
        help="Use per-point colors if available, or force distinct colors",
    )
    return parser


def main():
    args = make_parser().parse_args()

    if args.ref is None or args.src is None or args.corr is None:
        raise ValueError("Reference, source, and correspondence files are required")

    ref_points, ref_hsv, ref_rgb = load_point_cloud(
        args.ref, point_limit=args.point_limit
    )
    src_points, src_hsv, src_rgb = load_point_cloud(
        args.src, point_limit=args.point_limit
    )
    transform = load_transform(args.transform)

    with open(args.corr, "rb") as f:
        corr = pickle.load(f)
        estimated_transform = corr["estimated_transform"]
        ref_corr_points = corr["ref_corr_points"]
        src_corr_points = corr["src_corr_points"]

    visualize_pair(
        ref_points,
        src_points,
        estimated_transform,
        None,
        ref_corr_points,
        src_corr_points,
        args.vis_max_corr,
        ref_rgb,
        src_rgb,
        args.vis_color_mode == "raw",
    )


if __name__ == "__main__":
    main()
