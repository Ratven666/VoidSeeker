import numpy as np
from tqdm import tqdm

from app.mesh.utils.DownwardRayTracer import DownwardRayTracer

from app.voxel.Voxel import Voxel
from app.voxel.VoxelModelOpen3DVisualizer import VoxelModelOpen3DVisualizer
from app.voxel.VoxelModelPlotter import VoxelModelPlotter
from app.voxel.VoxelModelPyvistaPlotter import VoxelModelPyvistaPlotter
from app.voxel.VoxelPositionChecker import VoxelPositionChecker

class VoxelModel:
    def __init__(self, mesh, voxel_side, start_side, ray_tracer_class, refinement_factor=2):
        """
        mesh: Open3D TriangleMesh (или mesh.mesh, если оболочка)
        start_side: начальный крупный размер вокселя (грубая сетка)
        voxel_side: целевой финальный размер (до какого сгущать)
        ray_tracer_class: ваш HorizontalRayTracer (класс, а не экземпляр)
        refinement_factor: во сколько раз уменьшать шаг на каждом этапе
        """
        self.mesh = mesh
        bbox = mesh.mesh.get_axis_aligned_bounding_box()
        self.min_bound = bbox.min_bound
        self.max_bound = bbox.max_bound
        self.start_side = start_side
        self.voxel_side = voxel_side
        self.ray_tracer = ray_tracer_class(self.mesh)
        self.voxels = []
        valid_voxels = self._build_initial()
        self.voxels = self.refine_voxels(valid_voxels, self.voxel_side, refinement_factor)

    def _build_initial(self):
        # Вычисляем число слоёв явно
        nx = int(np.ceil((self.max_bound[0] - self.min_bound[0]) / self.start_side)) + 1
        ny = int(np.ceil((self.max_bound[1] - self.min_bound[1]) / self.start_side)) + 1
        nz = int(np.ceil((self.max_bound[2] - self.min_bound[2]) / self.start_side)) + 1

        x_vals = np.linspace(self.min_bound[0], self.max_bound[0], nx)
        y_vals = np.linspace(self.min_bound[1], self.max_bound[1], ny)
        z_vals = np.linspace(self.min_bound[2], self.max_bound[2], nz)

        valid_voxels = []
        total = len(x_vals) * len(y_vals) * len(z_vals)
        with tqdm(total=total, desc="Initial coarse voxels", unit="voxels") as pbar:
            for x in x_vals:
                for y in y_vals:
                    for z in z_vals:
                        voxel = Voxel([x, y, z], self.start_side)
                        checker = VoxelPositionChecker(voxel, self.ray_tracer)
                        status = checker.voxel_status()
                        voxel.status = status
                        if status == "inside" or status == "on_surface":
                            valid_voxels.append(voxel)
                        pbar.update(1)
        return valid_voxels

    def refine_voxels(self, active_voxels, voxel_side, refinement_factor=2):
        curr_side = self.start_side
        stage = 1
        while curr_side > voxel_side:
            new_side = max(curr_side / refinement_factor, voxel_side)
            next_voxels = []
            desc = f"Refinement {stage}: {curr_side:.4f}->{new_side:.4f}"
            with tqdm(total=len(active_voxels), desc=desc, unit="groups") as pbar:
                for voxel in active_voxels:
                    min_corner = voxel.min_corner
                    for dx in np.arange(0, curr_side, new_side):
                        for dy in np.arange(0, curr_side, new_side):
                            for dz in np.arange(0, curr_side, new_side):
                                small_corner = min_corner + np.array([dx, dy, dz])
                                small_voxel = Voxel(small_corner, new_side)
                                checker = VoxelPositionChecker(small_voxel, self.ray_tracer)
                                status = checker.voxel_status()
                                small_voxel.status = status
                                if status == "inside" or status == "on_surface":
                                    next_voxels.append(small_voxel)
                    pbar.update(1)
            curr_side = new_side
            active_voxels = next_voxels
            stage += 1
        return active_voxels

    def get_voxels(self):
        return self.voxels

    def plot(self, plotter=VoxelModelPlotter):
        plotter = plotter(self)
        plotter.plot()

    def __len__(self):
        return len(self.voxels)

if __name__ == "__main__":
    from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
    from app.mesh.mesh_trimmers.ZLevelsMeshTrimmer import ZLevelsMeshTrimmer
    from app.scan.Scan import Scan


    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../src/Камеры/data_3.dxf")

    mesh = PoissonMeshOpen3D(depth=12,
                             width=0.5,
                             scale=1.1).init_mesh_from_scan(scan)
    print(mesh)
    # mesh.plot()
    z_level = scan.borders["z_max"]
    print(z_level)
    mesh = ZLevelsMeshTrimmer(base_mesh=mesh, z_level=z_level).trim_mesh()
    # mesh.plot()

    vm = VoxelModel(mesh, voxel_side=1, start_side=1, ray_tracer_class=DownwardRayTracer, refinement_factor=2)

    # for voxel in vm.get_voxels():
    #     print(voxel)

    vm.plot(plotter=VoxelModelPyvistaPlotter)
