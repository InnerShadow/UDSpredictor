
from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

class CombineDrawer(DrawingSystem):

    def add_list(self, pathes : list[str]):
        pass

    def add_single_path(self, 
                    path : str):
        pass
    # end def

    def save(self, path: str) -> None:
        pass
    # end def

    def plot(self) -> None:
        pass
    # end def
# end class
