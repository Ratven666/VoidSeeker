import numpy as np
import open3d as o3d

from app.mesh.mesh_trimmers.MeshTrimmerABC import MeshTrimmerABC


class ZLevelsMeshTrimmer(MeshTrimmerABC):

    def __init__(self, base_mesh, z_level, above=False):
        super().__init__(base_mesh)
        self.z_level = z_level
        self.above = above

    def _custom_trim_logic(self):
        """
        Внутренний метод для обрезки по уровню Z с точным совпадением вершин с высотой обрезки
        """
        if len(self.base_mesh.mesh.triangles) == 0:
            return self.base_mesh.mesh

        vertices = np.asarray(self.base_mesh.mesh.vertices)
        triangles = np.asarray(self.base_mesh.mesh.triangles)

        # Определяем маску вершин в зависимости от направления обрезки
        if self.above:
            # Сохраняем вершины ВЫШЕ z_level
            vertices_mask = vertices[:, 2] >= self.z_level
            mode_name = f"выше Z={self.z_level}"
        else:
            # Сохраняем вершины НИЖЕ z_level
            vertices_mask = vertices[:, 2] <= self.z_level
            mode_name = f"ниже Z={self.z_level}"

        # Списки для новых вершин и треугольников
        new_vertices = []
        new_triangles = []
        vertex_mapping = {}  # маппинг старых индексов к новым

        # Счетчик новых вершин
        new_vertex_count = 0

        # Обрабатываем каждый треугольник
        for triangle in triangles:
            triangle_vertices = vertices[triangle]
            triangle_z = triangle_vertices[:, 2]

            # Определяем, какие вершины удовлетворяют условию
            if self.above:
                valid_vertices = triangle_z >= self.z_level
            else:
                valid_vertices = triangle_z <= self.z_level

            valid_count = np.sum(valid_vertices)

            if valid_count == 3:
                # Все вершины удовлетворяют условию - сохраняем треугольник целиком
                new_triangle = []
                for i, vertex_idx in enumerate(triangle):
                    if vertex_idx not in vertex_mapping:
                        new_vertices.append(vertices[vertex_idx])
                        vertex_mapping[vertex_idx] = new_vertex_count
                        new_vertex_count += 1
                    new_triangle.append(vertex_mapping[vertex_idx])
                new_triangles.append(new_triangle)

            elif valid_count == 2:
                # Две вершины удовлетворяют условию, одна - нет
                # Находим индексы валидных и невалидных вершин
                valid_indices = np.where(valid_vertices)[0]
                invalid_idx = np.where(~valid_vertices)[0][0]

                # Создаем новые вершины на уровне обрезки для пересечений
                new_cut_vertices = []
                for valid_idx in valid_indices:
                    # Для каждой пары валидная-невалидная вершина находим точку пересечения
                    valid_vertex = triangle_vertices[valid_idx]
                    invalid_vertex = triangle_vertices[invalid_idx]

                    # Вычисляем параметр t для линейной интерполяции
                    if abs(valid_vertex[2] - invalid_vertex[2]) > 1e-10:
                        t = (self.z_level - invalid_vertex[2]) / (valid_vertex[2] - invalid_vertex[2])
                    else:
                        t = 0.5

                    # Интерполируем координаты
                    cut_vertex = invalid_vertex + t * (valid_vertex - invalid_vertex)
                    cut_vertex[2] = self.z_level  # Точное совпадение с уровнем обрезки

                    new_vertices.append(cut_vertex)
                    new_cut_vertices.append(new_vertex_count)
                    new_vertex_count += 1

                # Создаем новые треугольники
                # Добавляем валидные вершины
                for valid_idx in valid_indices:
                    vertex_idx = triangle[valid_idx]
                    if vertex_idx not in vertex_mapping:
                        new_vertices.append(vertices[vertex_idx])
                        vertex_mapping[vertex_idx] = new_vertex_count
                        new_vertex_count += 1

                # Создаем треугольник из двух валидных вершин и одной новой точки пересечения
                new_triangle1 = [
                    vertex_mapping[triangle[valid_indices[0]]],
                    vertex_mapping[triangle[valid_indices[1]]],
                    new_cut_vertices[0]
                ]
                new_triangles.append(new_triangle1)

                # Создаем второй треугольник для сохранения формы
                new_triangle2 = [
                    vertex_mapping[triangle[valid_indices[1]]],
                    new_cut_vertices[1],
                    new_cut_vertices[0]
                ]
                new_triangles.append(new_triangle2)

            elif valid_count == 1:
                # Одна вершина удовлетворяет условию, две - нет
                valid_idx = np.where(valid_vertices)[0][0]
                invalid_indices = np.where(~valid_vertices)[0]

                # Создаем новые вершины на уровне обрезки
                new_cut_vertices = []
                for invalid_idx in invalid_indices:
                    valid_vertex = triangle_vertices[valid_idx]
                    invalid_vertex = triangle_vertices[invalid_idx]

                    # Вычисляем параметр t для линейной интерполяции
                    if abs(valid_vertex[2] - invalid_vertex[2]) > 1e-10:
                        t = (self.z_level - invalid_vertex[2]) / (valid_vertex[2] - invalid_vertex[2])
                    else:
                        t = 0.5

                    # Интерполируем координаты
                    cut_vertex = invalid_vertex + t * (valid_vertex - invalid_vertex)
                    cut_vertex[2] = self.z_level  # Точное совпадение с уровнем обрезки

                    new_vertices.append(cut_vertex)
                    new_cut_vertices.append(new_vertex_count)
                    new_vertex_count += 1

                # Добавляем валидную вершину
                vertex_idx = triangle[valid_idx]
                if vertex_idx not in vertex_mapping:
                    new_vertices.append(vertices[vertex_idx])
                    vertex_mapping[vertex_idx] = new_vertex_count
                    new_vertex_count += 1

                # Создаем треугольник из валидной вершины и двух точек пересечения
                new_triangle = [
                    vertex_mapping[vertex_idx],
                    new_cut_vertices[0],
                    new_cut_vertices[1]
                ]
                new_triangles.append(new_triangle)

        # Создаем новую сетку
        trimmed_mesh = o3d.geometry.TriangleMesh()

        if len(new_triangles) == 0:
            print(f"Предупреждение: Все треугольники удалены при обрезке {mode_name}")
            return o3d.geometry.TriangleMesh()

        # Устанавливаем вершины и треугольники
        trimmed_mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
        trimmed_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))

        # Вычисляем нормали для новой геометрии
        trimmed_mesh.compute_vertex_normals()
        trimmed_mesh.compute_triangle_normals()

        # Копируем цвета вершин, если они есть (только для оригинальных вершин)
        if (self.base_mesh.mesh.has_vertex_colors() and
                len(self.base_mesh.mesh.vertex_colors) == len(vertices)):

            # Создаем массив цветов для новых вершин
            new_colors = []
            for i, vertex in enumerate(new_vertices):
                # Для новых вершин на уровне обрезки используем средний цвет
                # или цвет ближайшей оригинальной вершины
                if i < len(vertices):  # оригинальные вершины
                    new_colors.append(self.base_mesh.mesh.vertex_colors[i])
                else:
                    # Для сгенерированных вершин используем серый цвет
                    new_colors.append([0.7, 0.7, 0.7])

            trimmed_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(new_colors))

        print(f"Обрезка по Z-уровню ({mode_name}):")
        print(f"  Треугольники: {len(new_triangles)}/{len(triangles)} создано")
        print(f"  Вершины: {len(new_vertices)}/{len(vertices)} создано")
        print(f"  Уровень обрезки: Z={self.z_level:.3f}")

        # Проверяем точность совпадения с уровнем обрезки
        new_vertices_array = np.asarray(trimmed_mesh.vertices)
        cut_vertices_z = new_vertices_array[:, 2]
        precision = np.max(np.abs(cut_vertices_z - self.z_level))
        print(f"  Точность совпадения с уровнем обрезки: {precision:.6f}")

        return trimmed_mesh


if __name__ == "__main__":
    from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
    from app.scan.Scan import Scan

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../../src/Камеры/data_3.dxf")

    mesh = PoissonMeshOpen3D(depth=12,
                             width=0,
                             scale=1.1).init_mesh_from_scan(scan)
    print(mesh)
    # mesh.plot()
    z_level = scan.borders["z_max"]
    print(z_level)
    mesh = ZLevelsMeshTrimmer(base_mesh=mesh, z_level=z_level).trim_mesh()
    mesh.plot()