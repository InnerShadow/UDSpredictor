import csv
import os

from typing import Callable
import torch
from torch import Tensor
from torch.nn import MSELoss, Module
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

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
                 output_size : int = 1,
                 use_batch_norm : bool = False,
                 dropout_rate : float = 0.0,
                 log_path: str = 'Outputs/data.csv'
                 ) -> None:
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.rnn_type = rnn_type
        self.optimazer = optimazer
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.log_path = log_path

        os.makedirs(os.path.dirname(self.log_path), exist_ok = True)
        self.first_epoch: bool = True
        
        self.model : Module = self._create_model()
    # end def

    def _log_to_csv(self, epoch: int, train_loss: float, val_loss: float) -> None:
        mode = 'w' if self.first_epoch else 'a'
        with open(self.log_path, mode, newline = '') as file:
            writer = csv.writer(file)
            if self.first_epoch:
                writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
                self.first_epoch = False

            val_loss_value = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
            writer.writerow([epoch, train_loss, val_loss_value])
        # end with
    # end def
        
    def _create_model(self):

        args = {
            'input_size' : self.input_size,
            'hidden_size' : self.hidden_size,
            'output_size' : self.output_size,
            'num_layers' : self.num_layers,
            'seq_len' : self.seq_len,
            'use_batch_norm' : self.use_batch_norm,
            'dropout_rate' : self.dropout_rate
        }

        model_map = {
            'lstm' : LSTMModel(**args).to('cpu'),
            'gru' : GRUModel(**args).to('cpu'),
        }

        return model_map[self.rnn_type.lower()]
    # end def

    def fit(self, 
            X : Tensor,
            y : Tensor,
            X_val : Tensor = None,
            y_val : Tensor = None,
            epochs : int = 10,
            lr : float = 0.001,
            batch_size : int = 32,
            patience_early_stopping : int = 16,
            patience_lr_reduce : int = 7,
            lr_reduce_factor : float = 0.1,
            minmun_lr : float = 10e-7,
            verbose : bool = False
            ) -> None:
        
        dataset_train = TensorDataset(X, y)
        dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)

        optimizer = OPTIMAZER_MAP[self.optimazer.lower()](self.model.parameters(), lr = lr)

        best_loss = float('inf')
        epochs_no_improve = 0
        epochs_no_improve_lr = 0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for X_batch, y_batch in dataloader_train:
                optimizer.zero_grad()
                outputs : Tensor = self.model(X_batch)
                loss : Tensor = TorchModel.loss_func(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # end for

            avg_train_loss = running_loss / len(dataloader_train)

            avg_loss = running_loss / len(dataloader_train)

            if X_val is not None and y_val is not None:
                predictions = self.iterative_predict(X_val, steps = len(y_val))
                val_loss = TorchModel.loss_func(predictions, y_val)
            else:
                val_loss = avg_loss
            # end if

            if verbose:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss}, Val Loss: {val_loss}')
            # end if

            self._log_to_csv(epoch + 1, avg_train_loss, val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                epochs_no_improve_lr = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1
                epochs_no_improve_lr += 1
            # end def

            # Early stopping
            if epochs_no_improve >= patience_early_stopping:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}. Best val_loss: {best_loss}")
                # end if
                break
            # end if

            # Reduce learning rate
            if epochs_no_improve_lr >= patience_lr_reduce:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * lr_reduce_factor, minmun_lr)
                # end for
                epochs_no_improve_lr = 0
                if verbose:
                    print(f"Reduced learning rate to {param_group['lr']}")
                # end if
            # end if
        # end for

        # Load the best model state (from early stopping)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        # end if
    # end def


    def iterative_predict(self, X_init: Tensor, steps: int) -> Tensor:
        self.model.eval()
        predictions = []
        
        X = X_init.clone()

        for _ in range(steps):
            with torch.no_grad():
                output = self.model(X)

                if output.dim() == 2:
                    output = output.unsqueeze(1)
                
                predictions.append(output[:, -1:, :])  
                
                X = torch.cat((X[:, 1:, :], output[:, -1:, :]), dim=1)
            # end with
        # end for
    
        return torch.cat(predictions, dim=1)
    # end def

    def score(self, X : Tensor, y : Tensor) -> float:
        self.model.eval()
    
        steps = y.size(1)
        
        with torch.no_grad():
            outputs: Tensor = self.iterative_predict(X, steps)
            mse: float = TorchModel.loss_func(outputs, y).item()
        
        return mse
    # end def

    def predict(self, X: Tensor, steps: int) -> Tensor:
        self.model.eval()
        
        with torch.no_grad():
            predictions: Tensor = self.iterative_predict(X, steps)
            predictions = predictions[:, -1, :]
            
        return predictions
    # end def
# end class
