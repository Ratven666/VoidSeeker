import open3d as o3d
import numpy as np
from app.mesh.MeshOpen3DABC import MeshOpen3DABC

class MeshDxf(MeshOpen3DABC):
    """
    Класс для загрузки полигональной поверхности из DXF (Polyface Mesh)
    в структуру MeshOpen3DABC без изменений геометрии.
    """
    def __init__(self):
        super().__init__()

    def _calculate_mesh(self):
        """
        Реализация абстрактного метода. 
        В данном случае mesh уже загружен, просто возвращаем его.
        """
        return self.mesh

    def init_mesh_from_dxf(self, file_path: str):
        """
        Парсит DXF файл, извлекает вершины и грани, создает Open3D TriangleMesh.
        """
        vertices, faces = self._parse_dxf_polyface(file_path)

        if not vertices or not faces:
            raise ValueError(f"Не удалось найти геометрию в файле {file_path}")

        # Создаем Open3D mesh
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64))
        self.mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
        
        # Для корректной работы базового класса создаем облако точек из вершин меша
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = self.mesh.vertices
        
        # Опционально: вычисляем нормали для корректного отображения (шейдинга)
        # self.mesh.compute_vertex_normals() 
        
        return self

    def _parse_dxf_polyface(self, filename: str):
        """
        Кастомный парсер для Polyface Mesh структуры DXF.
        Извлекает координаты (VERTEX с кодами 10,20,30) и индексы граней (VERTEX с кодами 71-74).
        """
        vertices = []
        faces = []
        
        try:
            with open(filename, 'r', encoding='cp1251', errors='ignore') as f:
                lines = [l.strip() for l in f]
        except UnicodeDecodeError:
            # Fallback для utf-8 если cp1251 не сработает
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [l.strip() for l in f]

        in_vertex = False
        current_data = {}
        
        # Простой конечный автомат для парсинга пар "Код-Значение"
        i = 0
        while i < len(lines) - 1:
            code = lines[i]
            val = lines[i+1]
            i += 2
            
            if code == '0':  # Начало новой сущности
                if in_vertex:
                    # Обрабатываем предыдущую вершину
                    self._process_vertex_data(current_data, vertices, faces)
                
                if val == 'VERTEX':
                    in_vertex = True
                    current_data = {}
                else:
                    in_vertex = False
            
            elif in_vertex:
                current_data[code] = val
        
        # Обработка последнего элемента
        if in_vertex:
            self._process_vertex_data(current_data, vertices, faces)
            
        return vertices, faces

    def _process_vertex_data(self, data, vertices, faces):
        """Вспомогательный метод для разбора данных одной вершины"""
        # Проверяем наличие индексов граней (Polyface mesh face definition)
        # Коды 71, 72, 73, 74 хранят индексы вершин
        face_indices = []
        is_face_def = False
        
        for k in ['71', '72', '73', '74']:
            if k in data:
                idx = int(data[k])
                if idx != 0:
                    # DXF индексы 1-based, и отрицательные означают скрытые ребра.
                    # Нам нужны абсолютные значения и конвертация в 0-based.
                    face_indices.append(abs(idx) - 1)
                    is_face_def = True
        
        if is_face_def:
            # Это определение грани
            if len(face_indices) >= 3:
                # Если это 4-угольник, Open3D (и STL) требует треугольники.
                # Разобьем 4-угольник на 2 треугольника: [0,1,2] и [0,2,3]
                if len(face_indices) == 4:
                    faces.append([face_indices[0], face_indices[1], face_indices[2]])
                    faces.append([face_indices[0], face_indices[2], face_indices[3]])
                else:
                    faces.append(face_indices)
        else:
            # Это геометрическая вершина (должна иметь координаты 10, 20, 30)
            if '10' in data:
                x = float(data.get('10', 0))
                y = float(data.get('20', 0))
                z = float(data.get('30', 0))
                vertices.append([x, y, z])


if __name__ == '__main__':
    mesh = MeshDxf().init_mesh_from_dxf(file_path=r"../../src/Рудное тело.dxf")
    mesh.plot()