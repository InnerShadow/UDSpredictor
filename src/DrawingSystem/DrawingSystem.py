
from pandas import DataFrame
from abc import ABC, abstractmethod

class DrawingSystem(ABC):
    def __init__(self) -> None:
        self.list_of_data : list[DataFrame] = []
        self.list_of_colors : list[str] = []
    # end def

    @abstractmethod
    def save(self, path : str) -> None:
        raise NotImplementedError()
    # end def

    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError()
    # end def
# end class
