
from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

import os

import matplotlib.pyplot as plt

# from sktime.utils.plotting import plot_series

class ForecastDrawer(DrawingSystem):
    def __init__(self) -> None:
        super().__init__()
        self.known_data : ndarray = ndarray()
        self.forecast_data : ndarray = ndarray()
    # end def

    def add_known_data(self, 
                    x : ndarray, 
                    y : ndarray, 
                    label : str,
                    color : str = 'Next') -> None:
        if color == 'Next':
            color = self.get_next_color()
        self.known_data.append((x, y, label, color))
    # end def

    def add_forecast_data(self, 
                    x_true : ndarray, 
                    y_true : ndarray,
                    true_label : str,
                    x : ndarray, 
                    y : ndarray, 
                    label : str,
                    color : str = 'Next') -> None:
        if color == 'Next':
            color = self.get_next_color()
        self.forecast_data.append((x_true, y_true, true_label, x, y, label, color))
    # end def

    # 
    def get_next_color(self) -> str:
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        index = len(self.list_of_colors) % len(colors)
        color = colors[index]
        self.list_of_colors.append(color)
        return(color)
    # end def

    def save(self, path: str) -> None:
        if self.known_data or self.forecast_data:
            filepath = os.path.join(BASE_OUTPUT_PATH, path)
            plt.savefig(filepath)
        else:
            raise ValueError('No data to save')
    # end def

    def plot(self) -> None:
        plt.figure()

        for x, y, label, color in self.known_data:
            plt.plot(x, y, label = label, color = color)
            # plot_series(y, label = label, ax = plt.gca(), color ='blue')

        for x_true, y_true, true_label, x, y, label, color in self.forecast_data:
            plt.plot(x_true, y_true, label = true_label, color = color)
            plt.plot(x, y, label = label, color = color)

            # plot_series(y_true, label = true_label, ax = plt.gca(), color = 'green')
            # plot_series(y, label = label, ax = plt.gca(), color = 'orange')

        plt.title('Forecast and Known Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    # end def
# end class
