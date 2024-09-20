
from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

class MultiChartDrawer(DrawingSystem):
    def add_new_chart(self, 
                      x : ndarray, 
                      y : ndarray, 
                      label : str,
                      color : str = 'Next'):
        pass
    # end def

    def save(self, path: str) -> None:
        pass
    # end def

    def plot(self) -> None:
        pass
    # end def
# end class
