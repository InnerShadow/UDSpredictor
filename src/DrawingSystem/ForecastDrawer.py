
from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

class ForecastDrawer(DrawingSystem):

    def add_known_data(self, 
                    x : ndarray, 
                    y : ndarray, 
                    label : str,
                    color : str = 'Next'):
        pass
    # end def

    def add_forecast_data(self, 
                    x_true : ndarray, 
                    y_true : ndarray,
                    true_label : str,
                    x : ndarray, 
                    y : ndarray, 
                    label : str,
                    color : str = 'Next'):
        pass
    # end def

    def save(self, path: str) -> None:
        pass
    # end def

    def plot(self) -> None:
        pass
    # end def
# end class
