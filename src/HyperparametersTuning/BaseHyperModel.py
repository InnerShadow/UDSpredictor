
import numpy as np

from abc import ABC, abstractmethod

class BaseHyperModel(ABC):

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError()
    # end def

    @abstractmethod
    def get_params(self, deep = True) -> dict:
        raise NotImplementedError()
    # end def

    def set_params(self, **params) -> None:
        for param, value in params.items():
            setattr(self, param, value)
        # end for
        return self
    # end def

    @abstractmethod
    def fit(self, X : np.ndarray, y : np.ndarray) -> None:
        raise NotImplementedError()
    # end def

    @abstractmethod
    def score(self, X : np.ndarray, y : np.ndarray) -> np.float32:
        raise NotImplementedError()
    # end def
# end class
