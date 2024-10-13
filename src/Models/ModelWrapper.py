
from numpy import ndarray

from typing import Optional

from Models.TorchModel import TorchModel
from Models.BaseModel import BaseModel

class ModelWrapper(object):
    def __init__(self, type : str, **kwargs):

        self.model : BaseModel = None

        match type.lower():
            case 'lstm':
                self.model = TorchModel(**kwargs, rnn_type = 'lstm')
            # end case
            case 'gru':
                self.model = TorchModel(**kwargs, rnn_type = 'gru')
            # end case
        # end match
    # end def

    def fit(self, **kwargs) -> None:
        return self.model.fit(**kwargs)
    # end def

    def score(self, **kwargs) -> float:
        return self.model.score(**kwargs)
    # end def

    def predict(self, X : Optional[ndarray], **kwargs) -> Optional[list]:
        return self.model.predict(X, **kwargs)
    # end def
# end class
