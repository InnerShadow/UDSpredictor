
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from abc import ABC, abstractmethod

class DataLoader(ABC):
    def __init__(self,
                 do_scale : bool = False) -> None:
        self.df : DataFrame = None
        self._preprocess_data_frame()
        self._load_data()
        if do_scale:
            self._do_scale()
        # end if
    # end def

    @abstractmethod
    def _preprocess_data_frame(self) -> None:
        raise NotImplementedError()
    # end def

    def _load_data(self):
        pass
    # end def

    def __len__(self) -> int:
        pass
    # end def

    def __getitem__(self, index : int) -> float:
        pass
    # end def

    def _do_scale(self) -> None:
        pass
    # end def

# end class
