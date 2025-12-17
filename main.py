from ezdxf.entities import Mesh

from app.VolumeCalculator import VolumeCalculator
from app.mesh.MeshDxf import MeshDxf
from app.mesh.MeshOpen3DABC import MeshOpen3DABC
from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
from app.mesh.mesh_trimmers.BBoxMeshTrimmer import BBoxMeshTrimmer
from app.mesh.mesh_trimmers.ZLevelsMeshTrimmer import ZLevelsMeshTrimmer
from app.mesh.utils.VectorRayTracer import VectorRayTracer
from app.scan.Scan import Scan
from app.voxel.VoxelModel import VoxelModel

scan = Scan(scan_name="TestScan")
scan.import_points_from_file(file_path="src/Камеры/data_3.dxf")
# scan.plot()
print(scan)

mesh_camera = PoissonMeshOpen3D(depth=12,
                                width=0.5,
                                scale=1.1).init_mesh_from_scan(scan)
z_level = scan.borders["z_max"]
mesh_camera = ZLevelsMeshTrimmer(base_mesh=mesh_camera, z_level=z_level).trim_mesh()

mesh_ruda = MeshDxf().init_mesh_from_dxf("src/Рудное тело.dxf")


bbox = mesh_camera.mesh.get_axis_aligned_bounding_box()
print(bbox)
min_bound = bbox.min_bound
max_bound = bbox.max_bound
print(min_bound, max_bound)
# # Определяем bbox для обрез
bbox_bounds = (
    (min_bound[0] - 5, max_bound[0] + 5),  # x_min, x_max
    (min_bound[1] - 5, max_bound[1] + 5),  # y_min, y_max
    (min_bound[2] - 5, max_bound[2] + 5)  # z_min, z_max
)
print(bbox_bounds)

# Обрезаем поверхность с заполнением граничных граней
trimmer = BBoxMeshTrimmer(
    base_mesh=mesh_ruda,
    bbox_bounds=bbox_bounds,
    fill_borders=True,
)

mesh_ruda = trimmer.trim_mesh()

# import open3d as o3d
# o3d.visualization.draw_geometries([mesh_ruda.pcd])
# Визуализируем
# mesh_ruda.plot()

# vm_ruda = VoxelModel(mesh_ruda, voxel_side=0.5, start_side=2, refinement_factor=2,
#                      ray_tracer_class=VectorRayTracer,
#                      )
# print(vm_ruda)
# # vm_ruda.plot()
# #
# #
# # # mesh.plot()
# #
# vm_camera = VoxelModel(mesh_camera, voxel_side=0.5, start_side=2, refinement_factor=2)
# # vm_camera.plot()
#
# vm_volume = vm_ruda.get_difference_between_vm(vm_camera)
# # vm_volume = vm_camera.get_difference_between_vm(vm_ruda)
#
# volume = VolumeCalculator(voxel_model=vm_camera).calculate_volume()
# print(f"Volume: {volume}")
# volume = VolumeCalculator(voxel_model=vm_volume).calculate_volume()
# print(f"Volume: {volume}")
#
# # vm_volume.plot()
