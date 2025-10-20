from abc import ABC, abstractmethod


class MeshPlotterABC(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot(self, mesh):
        pass
