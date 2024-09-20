
from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

class ResidueDrawer(DrawingSystem):
    def add_residue(self, 
                    x_true : ndarray, 
                    x_predicted : ndarray, 
                    cmap : str = 'viridis'):
        pass
    # end def

    def save(self, path: str) -> None:
        pass
    # end def

    def plot(self) -> None:
        pass
    # end def
# end class
