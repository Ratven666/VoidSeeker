from app.voxel.VoxelModel import VoxelModel


class VolumeCalculator:

    def __init__(self, voxel_model: VoxelModel):
        self.voxel_model = voxel_model

    def calculate_volume(self):
        volume = 0
        base_volume = self.voxel_model.voxels[0].size ** 3
        for voxel in self.voxel_model:
            if voxel.status == "inside":
                volume += base_volume
            elif voxel.status == "on_surface":
                volume += (base_volume / 2)
        return volume

if __name__ == "__main__":
    from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
    from app.mesh.mesh_trimmers.ZLevelsMeshTrimmer import ZLevelsMeshTrimmer
    from app.scan.Scan import Scan

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../src/Камеры/data_3.dxf")

    mesh = PoissonMeshOpen3D(depth=12,
                             width=0.5,
                             scale=1.1).init_mesh_from_scan(scan)

    z_level = scan.borders["z_max"]
    mesh = ZLevelsMeshTrimmer(base_mesh=mesh, z_level=z_level).trim_mesh()

    vm = VoxelModel(mesh, voxel_side=0.125, start_side=1, refinement_factor=2)
    print(vm)

    volume = VolumeCalculator(voxel_model=vm).calculate_volume()
    print(f"Volume: {volume}")