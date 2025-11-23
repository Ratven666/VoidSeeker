from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
from app.mesh.mesh_trimmers.ZLevelsMeshTrimmer import ZLevelsMeshTrimmer
from app.scan.Scan import Scan

scan = Scan(scan_name="TestScan")
scan.import_points_from_file(file_path="src/Камеры/data_3.dxf")

mesh = PoissonMeshOpen3D(depth=12,
                         width=0.5,
                         scale=1.1).init_mesh_from_scan(scan)
print(mesh)
# mesh.plot()
z_level = scan.borders["z_max"]
print(z_level)
mesh = ZLevelsMeshTrimmer(base_mesh=mesh, z_level=z_level).trim_mesh()
mesh.plot()