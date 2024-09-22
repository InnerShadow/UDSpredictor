
from typing import Callable

import torch
from torch import Tensor
from torch.nn import MSELoss

from Models.BaseModel import BaseModel
from Models.LSTMModel import LSTMModel
from Models.GRUModel import GRUModel

from Config.config import OPTIMAZER_MAP

class TorchModel(BaseModel):
    loss_func: Callable[[Tensor, Tensor], Tensor] = MSELoss()

    def __init__(self, 
                 hidden_size : int, 
                 num_layers : int, 
                 seq_len : int,
                 rnn_type : str,
                 optimazer : str,
                 input_size : int = 1,
                 output_size : int = 1
                 ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.rnn_type = rnn_type
        self.optimazer = optimazer
        
        self.model = self._create_model()
    # end def
        
    def _create_model(self):

        model_map = {
            'lstm' : LSTMModel(input_size = self.input_size, 
                        hidden_size = self.hidden_size, 
                        output_size = self.output_size, 
                        num_layers = self.num_layers, 
                        seq_len = self.seq_len
                        ).to('cpu'),
            'gru' : GRUModel(input_size = self.input_size, 
                        hidden_size = self.hidden_size, 
                        output_size = self.output_size, 
                        num_layers = self.num_layers, 
                        seq_len = self.seq_len
                        ).to('cpu'),
        }

        return model_map[self.rnn_type.lower()]
    # end def

    def fit(self, 
            X : Tensor,
            y : Tensor,
            epochs : int = 10,
            lr : float = 0.001,
            verbose : bool = False
            ) -> None:
        
        optimizer = OPTIMAZER_MAP[self.optimazer.lower()](self.model.parameters(), lr = lr)

        for epoch in range(epochs):
            self.model.train()
            
            optimizer.zero_grad()
            outputs: Tensor = self.model(X)
            loss: Tensor = TorchModel.loss_func(outputs, y)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0 and verbose:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
            # end if
        # end for
    # end def

    def score(self, X: Tensor, y: Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            outputs: Tensor = self.model(X)
            mse: float = TorchModel.loss_func(outputs, y).item()
        # end with
        return mse
    # end def

    def predict(self, X: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            outputs: Tensor = self.model(X)
        # end with
        return outputs
    # end def
# end class
