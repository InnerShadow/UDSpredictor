import json
import os

import numpy as np
import torch

from HyperparametersTuning.BaseHyperModel import BaseHyperModel
from Models.ModelWrapper import ModelWrapper
from DataLoaders.DataLoaderWrapper import DataLoaderWrapper


class RNNHyperModel(BaseHyperModel):
    def __init__(self, 
                 do_scale: bool = True,
                 type: str = 'gru',
                 hidden_size: int = 10,
                 num_layers: int = 2,
                 seq_len: int = 20,
                 optimazer: str = 'adam',
                 use_batch_norm: bool = True,
                 dropout_rate: float = 0
                 ):
        
        self.do_scale = do_scale
        self.type = type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.optimazer = optimazer
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        self.run_number_file = 'Output/rnn_run_number.json'
        self.output_file = 'Output/rnn_tuning.json'
        self.ensure_output_files_exist()
        
        self.run_number = self.get_current_run_number()
    # end def

    def ensure_output_files_exist(self):
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as f:
                json.dump([], f)
            # end with
        # end if
        if not os.path.exists(self.run_number_file):
            with open(self.run_number_file, 'w') as f:
                json.dump({'run_number': 1}, f)
            # end with
        # end if

        with open(self.run_number_file, 'r') as f:
            data = json.load(f)
        # end with

        if data['run_number'] != 1:
            with open(self.run_number_file, 'w') as f:
                json.dump({'run_number': 1}, f)
            # end with
        # end if
    # end def

    def get_current_run_number(self):
        with open(self.run_number_file, 'r') as f:
            data = json.load(f)
        # end with
        return data['run_number']
    # end def

    def increment_run_number(self):
        self.run_number += 1
        if self.run_number > 3:
            self.run_number = 1
        with open(self.run_number_file, 'w') as f:
            json.dump({'run_number': self.run_number}, f)
        # end with
    # end def

    def save_result(self, params, score):
        data = self.load_results()
        data.append({
            'params': params,
            'run_number': self.run_number,
            'score': score
        })
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=6)
        # end with
    # end def

    def load_results(self):
        with open(self.output_file, 'r') as f:
            return json.load(f)
        # end with
    # end def

    def get_params(self, deep = True) -> dict:
        return {
            'do_scale': self.do_scale,
            'type': self.type,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'seq_len': self.seq_len,
            'optimazer': self.optimazer,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate
        }
    # end def

    def check_if_trained(self):
        params = self.get_params()
        results = self.load_results()
        for result in results:
            if result['params'] == params and result['run_number'] == self.run_number:
                return result['score']
        return None
    # end def

    def fit(self, X=0, y=0) -> None:
        existing_score = self.check_if_trained()
        if existing_score is not None:
            print(existing_score)
            return
        # end if

        print(self.get_params())

        self.model = ModelWrapper(
            type=self.type,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_len=self.seq_len,
            optimazer=self.optimazer,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate
        )
        
        X = []
        y = []

        data_loader = DataLoaderWrapper(period='post', do_scale=self.do_scale)
        data = data_loader.get_data()

        cost = data['Cost'].to_list()

        for i in range(len(cost) - self.seq_len):
            X.append(cost[i: i + self.seq_len])
            y.append(cost[i + self.seq_len])
        # end for

        train_scale = int(len(X) * 0.8)

        batch_size = 32
        epochs = 500

        X_train = torch.tensor(X[:train_scale], dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y[:train_scale], dtype=torch.float32).unsqueeze(-1)

        self.X_test = torch.tensor(X[train_scale:], dtype=torch.float32).unsqueeze(-1)
        self.y_test = torch.tensor(y[train_scale:], dtype=torch.float32).unsqueeze(-1)

        self.model.fit(
            X=X_train, 
            y=y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=True,
            X_val=self.X_test,
            y_val=self.y_test
        )
    # end def

    def score(self, X = 0, y = 0) -> np.float32:
        existing_score = self.check_if_trained()
        if existing_score is not None:
            self.increment_run_number()
            return existing_score
        # end if

        score = self.model.score(X = self.X_test, y = self.y_test)
        self.save_result(self.get_params(), score)

        self.increment_run_number()
        return score
    # end def
# end class
