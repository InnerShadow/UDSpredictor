import pmdarima as pm
import json
import os

from Models.BaseModel import BaseModel

from typing import Optional
from Config.config import BASE_OUTPUT_PATH

class AutoARIMAModel(BaseModel):

    def __init__(self, 
                start_p : int = 1,
                max_p : int = 5, 
                start_q : int = 1, 
                max_q : int = 5, 
                seasonal : bool = True,
                m : int = 12,
                max_d : int = 2,
                max_D: int = 1,
                n_fits: int = 50,
                stepwise: bool = True,
                random_state: Optional[int] = 42,
                log_path: str = 'Output/arima_logs.json',
                params_path: str = 'Output/arima_params.json') -> None:

        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.seasonal = seasonal
        self.m = m
        self.max_d = max_d
        self.max_D = max_D
        self.n_fits = n_fits
        self.stepwise = stepwise
        self.random_state = random_state
        self.log_path = log_path
        self.params_path = params_path
        self.model = None
        self.logs = []
        self.best_params = {}
        
        os.makedirs(os.path.dirname(self.log_path), exist_ok = True)
    # end def

    def _log_to_json(self, content: dict, path: str) -> None:
        with open(path, 'w') as file:
            json.dump(content, file, indent = 4)
    # end def

    def _create_model(self):
        return pm.auto_arima(
            start_p = self.start_p, 
            max_p = self.max_p,
            start_q = self.start_q, 
            max_q = self.max_q,
            seasonal = self.seasonal, 
            m = self.m,
            max_d = self.max_d, 
            max_D = self.max_D,
            trace = True,  
            error_action = 'ignore',  
            suppress_warnings = True,  
            stepwise = self.stepwise,  
            random_state = self.random_state,
            n_fits = self.n_fits  
        )
    # end def

    def fit(self, y_train, verbose: bool = False):

        self.model = self._create_model()

        if verbose:
            print("Training Auto ARIMA model...")
        
        self.model.fit(y_train)

        self.logs.append(self.model.cv_results_)
        self.best_params = self.model.get_params()

        if verbose:
            print(f"Best ARIMA model parameters: {self.best_params}")

        self._log_to_json(self.logs, self.log_path)
        self._log_to_json(self.best_params, self.params_path)
    # end def

    def score(self, y_test) -> float:
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call `fit` before scoring.")
        
        return self.model.score(y_test)
    # end def

    def predict(self, n_periods: int):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call `fit` before predicting.")
        
        return self.model.predict(n_periods = n_periods)
    # end def
# end class