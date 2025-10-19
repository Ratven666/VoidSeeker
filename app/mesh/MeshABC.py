from abc import ABC, abstractmethod


class MeshABC(ABC):

    @abstractmethod
    def _calculate_mesh(self):
        pass
