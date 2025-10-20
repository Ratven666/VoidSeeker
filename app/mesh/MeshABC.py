from abc import ABC, abstractmethod

from app.mesh.mesh_plotters.BaseOpen3DMeshPlotter import BaseOpen3DMeshPlotter
from app.mesh.mesh_plotters.MeshPlotterPyVista import MeshPlotterPyVista


class MeshABC(ABC):

    @abstractmethod
    def _calculate_mesh(self):
        pass

    # def plot(self, *args, plotter=BaseOpen3DMeshPlotter, **kwargs):
    def plot(self, *args, plotter=MeshPlotterPyVista, **kwargs):
        plotter = plotter(*args, **kwargs)
        fig_ax = plotter.plot(mesh=self)
        return fig_ax
