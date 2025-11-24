import numpy as np
import open3d as o3d
from app.mesh.mesh_trimmers.MeshTrimmerABC import MeshTrimmerABC

class ZLevelsMeshTrimmer(MeshTrimmerABC):
    def __init__(self, base_mesh, z_level, above=False):
        super().__init__(base_mesh)
        self.z_level = z_level
        self.above = above
        self.metadata = {}  # здесь будут доп. параметры

    def _custom_trim_logic(self):
        if len(self.base_mesh.mesh.triangles) == 0:
            self.metadata = {}
            return self.base_mesh.mesh

        vertices = np.asarray(self.base_mesh.mesh.vertices)
        triangles = np.asarray(self.base_mesh.mesh.triangles)
        if self.above:
            vertices_mask = vertices[:, 2] >= self.z_level
            mode_name = f"выше Z={self.z_level}"
        else:
            vertices_mask = vertices[:, 2] <= self.z_level
            mode_name = f"ниже Z={self.z_level}"

        new_vertices = []
        new_triangles = []
        vertex_mapping = {}
        new_vertex_count = 0
        cut_indices = []

        for triangle in triangles:
            triangle_vertices = vertices[triangle]
            triangle_z = triangle_vertices[:, 2]

            if self.above:
                valid_vertices = triangle_z >= self.z_level
            else:
                valid_vertices = triangle_z <= self.z_level
            valid_count = np.sum(valid_vertices)

            if valid_count == 3:
                new_triangle = []
                for i, vertex_idx in enumerate(triangle):
                    if vertex_idx not in vertex_mapping:
                        new_vertices.append(vertices[vertex_idx])
                        vertex_mapping[vertex_idx] = new_vertex_count
                        new_vertex_count += 1
                    new_triangle.append(vertex_mapping[vertex_idx])
                new_triangles.append(new_triangle)

            elif valid_count == 2:
                valid_indices = np.where(valid_vertices)[0]
                invalid_idx = np.where(~valid_vertices)[0][0]
                new_cut_vertices = []
                for valid_idx in valid_indices:
                    valid_vertex = triangle_vertices[valid_idx]
                    invalid_vertex = triangle_vertices[invalid_idx]
                    if abs(valid_vertex[2] - invalid_vertex[2]) > 1e-10:
                        t = (self.z_level - invalid_vertex[2]) / (valid_vertex[2] - invalid_vertex[2])
                    else:
                        t = 0.5
                    cut_vertex = invalid_vertex + t * (valid_vertex - invalid_vertex)
                    cut_vertex[2] = self.z_level
                    new_vertices.append(cut_vertex)
                    cut_indices.append(new_vertex_count)
                    new_cut_vertices.append(new_vertex_count)
                    new_vertex_count += 1
                for valid_idx in valid_indices:
                    vertex_idx = triangle[valid_idx]
                    if vertex_idx not in vertex_mapping:
                        new_vertices.append(vertices[vertex_idx])
                        vertex_mapping[vertex_idx] = new_vertex_count
                        new_vertex_count += 1
                new_triangle1 = [
                    vertex_mapping[triangle[valid_indices[0]]],
                    vertex_mapping[triangle[valid_indices[1]]],
                    new_cut_vertices[0]
                ]
                new_triangles.append(new_triangle1)
                new_triangle2 = [
                    vertex_mapping[triangle[valid_indices[1]]],
                    new_cut_vertices[1],
                    new_cut_vertices[0]
                ]
                new_triangles.append(new_triangle2)

            elif valid_count == 1:
                valid_idx = np.where(valid_vertices)[0][0]
                invalid_indices = np.where(~valid_vertices)[0]
                new_cut_vertices = []
                for invalid_idx in invalid_indices:
                    valid_vertex = triangle_vertices[valid_idx]
                    invalid_vertex = triangle_vertices[invalid_idx]
                    if abs(valid_vertex[2] - invalid_vertex[2]) > 1e-10:
                        t = (self.z_level - invalid_vertex[2]) / (valid_vertex[2] - invalid_vertex[2])
                    else:
                        t = 0.5
                    cut_vertex = invalid_vertex + t * (valid_vertex - invalid_vertex)
                    cut_vertex[2] = self.z_level
                    new_vertices.append(cut_vertex)
                    cut_indices.append(new_vertex_count)
                    new_cut_vertices.append(new_vertex_count)
                    new_vertex_count += 1
                vertex_idx = triangle[valid_idx]
                if vertex_idx not in vertex_mapping:
                    new_vertices.append(vertices[vertex_idx])
                    vertex_mapping[vertex_idx] = new_vertex_count
                    new_vertex_count += 1
                new_triangle = [
                    vertex_mapping[vertex_idx],
                    new_cut_vertices[0],
                    new_cut_vertices[1]
                ]
                new_triangles.append(new_triangle)

        trimmed_mesh = o3d.geometry.TriangleMesh()
        if len(new_triangles) == 0:
            print(f"Предупреждение: Все треугольники удалены при обрезке {mode_name}")
            self.metadata = {}
            return o3d.geometry.TriangleMesh()

        trimmed_mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
        trimmed_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
        trimmed_mesh.compute_vertex_normals()
        trimmed_mesh.compute_triangle_normals()

        # Метаданные
        bbox = trimmed_mesh.get_axis_aligned_bounding_box()
        surface_area = trimmed_mesh.get_surface_area()
        try:
            volume = trimmed_mesh.get_volume()
        except Exception:
            volume = None

        # Цвета (если были)
        if (self.base_mesh.mesh.has_vertex_colors() and
            len(self.base_mesh.mesh.vertex_colors) == len(vertices)):
            new_colors = []
            for i, vertex in enumerate(new_vertices):
                if i < len(vertices):
                    new_colors.append(self.base_mesh.mesh.vertex_colors[i])
                else:
                    new_colors.append([0.7, 0.7, 0.7])
            trimmed_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(new_colors))

        # Точность именно по новым граничным точкам!
        if cut_indices:
            cut_vertices_z = np.array(new_vertices)[cut_indices, 2]
            precision = np.max(np.abs(cut_vertices_z - self.z_level))
        else:
            precision = None

        # Сохраняем метаданные для анализа/дальнейшей работы
        self.metadata = {
            "bbox": bbox,
            "surface_area": surface_area,
            "volume": volume,
            "precision": precision
        }

        print(f"Обрезка по Z-уровню ({mode_name}):")
        print(f" Треугольники: {len(new_triangles)}/{len(triangles)} создано")
        print(f" Вершины: {len(new_vertices)}/{len(vertices)} создано")
        print(f" Уровень обрезки: Z={self.z_level:.3f}")
        if precision is not None:
            print(f" Точность совпадения с уровнем обрезки (только новые пересечения): {precision:.6f}")
        else:
            print(" Нет новых граничных вершин; все точки лежат по одну сторону уровня")
        print(f" BBox min {bbox.min_bound}, max {bbox.max_bound}")
        print(f" Площадь поверхности: {surface_area:.4f}")
        if volume is not None:
            print(f" Примерный объём: {volume:.4f}")

        return trimmed_mesh

    def get_metadata(self):
        return self.metadata


if __name__ == "__main__":
    from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
    from app.scan.Scan import Scan

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../../src/Камеры/data_3.dxf")

    mesh = PoissonMeshOpen3D(depth=12,
                             width=0.5,
                             scale=1.1).init_mesh_from_scan(scan)
    print(mesh)
    # mesh.plot()
    z_level = scan.borders["z_max"]
    print(z_level)
    mesh = ZLevelsMeshTrimmer(base_mesh=mesh, z_level=z_level).trim_mesh()
    mesh.plot()