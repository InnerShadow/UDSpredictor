
import torch
from torch import Tensor
from torch.nn import Module, Linear, LSTM

class LSTMModel(Module):
    def __init__(self, 
                 input_size : int, 
                 hidden_size : int, 
                 output_size : int, 
                 num_layers : int, 
                 seq_len : int
                 ) -> None:
        
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first = True)
        
        self.fc = Linear(hidden_size, output_size)
    # end def
    
    def forward(self, x: Tensor) -> Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out
    # end def
# end class
