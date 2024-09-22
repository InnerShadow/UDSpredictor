
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError()
    # end def

    @abstractmethod
    def _create_model(self):
        raise NotImplementedError()
    # end def

    @abstractmethod
    def fit(self):
        raise NotImplementedError()
    # end def

    @abstractmethod
    def score(self):
        raise NotImplementedError()
    # end def

    @abstractmethod
    def predict(self):
        raise NotImplementedError()
    # end def
# end class
