import os

import numpy as np
from scipy.spatial import cKDTree


def _as_vec3_array(value):
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return array.reshape(-1, 3).copy()


def _normalize_rows(vectors):
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return vectors / norms


def _estimate_normals(points, k_neighbors=12):
    points = _as_vec3_array(points)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(points) < 3:
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0
        return normals

    k_neighbors = max(3, min(int(k_neighbors), len(points)))
    tree = cKDTree(points)
    _, neighbor_idx = tree.query(points, k=k_neighbors)
    if neighbor_idx.ndim == 1:
        neighbor_idx = neighbor_idx[:, None]

    neighbors = points[neighbor_idx]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariances = np.einsum("nki,nkj->nij", centered, centered)
    _, eigenvectors = np.linalg.eigh(covariances)
    normals = eigenvectors[:, :, 0]

    centroid = points.mean(axis=0, keepdims=True)
    outward = points - centroid
    flip_mask = np.einsum("ni,ni->n", normals, outward) < 0
    normals[flip_mask] *= -1
    return _normalize_rows(normals)


def _average_by_group(values, inverse_indices, counts):
    if values.size == 0:
        return values
    output = np.zeros((len(counts), values.shape[1]), dtype=np.float64)
    np.add.at(output, inverse_indices, values)
    output /= counts[:, None]
    return output


class _UtilityModule:
    @staticmethod
    def Vector3dVector(value):
        return _as_vec3_array(value)


class PointCloud:
    def __init__(self):
        self._points = np.zeros((0, 3), dtype=np.float64)
        self._colors = np.zeros((0, 3), dtype=np.float64)
        self._normals = np.zeros((0, 3), dtype=np.float64)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = _as_vec3_array(value)

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = _as_vec3_array(value)

    @property
    def normals(self):
        return self._normals

    @normals.setter
    def normals(self, value):
        self._normals = _as_vec3_array(value)

    def __iadd__(self, other):
        self._points = np.vstack([self._points, _as_vec3_array(other.points)])

        if len(self._colors) and len(other.colors):
            self._colors = np.vstack([self._colors, _as_vec3_array(other.colors)])
        else:
            self._colors = np.zeros((0, 3), dtype=np.float64)

        if len(self._normals) and len(other.normals):
            self._normals = np.vstack([self._normals, _as_vec3_array(other.normals)])
        else:
            self._normals = np.zeros((0, 3), dtype=np.float64)
        return self

    def transform(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        if len(self._points):
            homogeneous = np.hstack([self._points, np.ones((len(self._points), 1), dtype=np.float64)])
            self._points = (homogeneous @ matrix.T)[:, :3]
        if len(self._normals):
            rotation = matrix[:3, :3]
            self._normals = _normalize_rows(self._normals @ rotation.T)
        return self

    def estimate_normals(self):
        self._normals = _estimate_normals(self._points)
        return self

    def voxel_down_sample(self, voxel_size):
        voxel_size = float(voxel_size)
        output = PointCloud()
        if voxel_size <= 0 or len(self._points) == 0:
            output.points = self._points
            output.colors = self._colors
            output.normals = self._normals
            return output

        voxel_ids = np.floor(self._points / voxel_size).astype(np.int64)
        _, inverse_indices = np.unique(voxel_ids, axis=0, return_inverse=True)
        counts = np.bincount(inverse_indices)

        output.points = _average_by_group(self._points, inverse_indices, counts)
        if len(self._colors) == len(self._points):
            output.colors = _average_by_group(self._colors, inverse_indices, counts)
        if len(self._normals) == len(self._points):
            averaged_normals = _average_by_group(self._normals, inverse_indices, counts)
            output.normals = _normalize_rows(averaged_normals)
        return output


class _GeometryModule:
    PointCloud = PointCloud


def _write_point_cloud_ascii(path, point_cloud):
    points = _as_vec3_array(point_cloud.points)
    colors = _as_vec3_array(point_cloud.colors) if len(point_cloud.colors) == len(points) else None
    normals = _as_vec3_array(point_cloud.normals) if len(point_cloud.normals) == len(points) else None

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if normals is not None:
        header.extend(
            [
                "property float nx",
                "property float ny",
                "property float nz",
            ]
        )
    if colors is not None:
        header.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    header.append("end_header")

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.max(initial=0.0) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(np.round(colors), 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(header))
        file.write("\n")
        for idx, point in enumerate(points):
            values = [f"{point[0]:.8f}", f"{point[1]:.8f}", f"{point[2]:.8f}"]
            if normals is not None:
                normal = normals[idx]
                values.extend([f"{normal[0]:.8f}", f"{normal[1]:.8f}", f"{normal[2]:.8f}"])
            if colors is not None:
                color = colors[idx]
                values.extend([str(int(color[0])), str(int(color[1])), str(int(color[2]))])
            file.write(" ".join(values))
            file.write("\n")
    return True


class _IOModule:
    @staticmethod
    def write_point_cloud(path, point_cloud):
        return _write_point_cloud_ascii(path, point_cloud)


geometry = _GeometryModule()
utility = _UtilityModule()
io = _IOModule()
