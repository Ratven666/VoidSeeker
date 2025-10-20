from abc import ABC, abstractmethod
from copy import deepcopy


class MeshTrimmerABC(ABC):

    def __init__(self, base_mesh):
        self.base_mesh = base_mesh

    def trim_mesh(self, inplace=True):
        trimmed_mesh = self._custom_trim_logic()
        if inplace:
            self.base_mesh.mesh = trimmed_mesh
        else:
            mesh_copy = deepcopy(self.base_mesh)
            mesh_copy.mesh = trimmed_mesh
            return mesh_copy
        return self.base_mesh

    @abstractmethod
    def _custom_trim_logic(self):
        pass
