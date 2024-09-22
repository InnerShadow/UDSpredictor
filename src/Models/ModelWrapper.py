
from numpy import ndarray

from typing import Optional

from Models.TorchModel import TorchModel

class ModelWrapper(object):
    def __init__(self, type : str, **kwargs):
        match type.lower():
            case 'lstm':
                self.model = TorchModel(**kwargs, rnn_type = 'lstm')
            # enc case
            case 'gru':
                self.model = TorchModel(**kwargs, rnn_type = 'lstm')
            # end case
        # end match
    # end def

    def fit(self, X : Optional[ndarray], 
              y : Optional[ndarray],
              **kwargs
              ) -> None:
        return self.model.fit(X = X, y = y, **kwargs)
    # end def

    def score(self, X : Optional[ndarray], 
              y : Optional[ndarray],
              ) -> float:
        return self.model.score(X = X, y = y)
    # end def

    def predict(self, X : Optional[ndarray]) -> Optional[list]:
        return self.model.predict(X)
    # end def
# end class
