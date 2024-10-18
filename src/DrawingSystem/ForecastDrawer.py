
from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

import os

import matplotlib.pyplot as plt


class ForecastDrawer(DrawingSystem):
    def __init__(self) -> None:
        super().__init__()
        self.known_data : ndarray = [] #ndarray()
        self.forecast_data : ndarray = [] #ndarray()
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
                    true_color : str = 'Next',
                    color = 'Next') -> None:
        if true_color == 'Next':
            true_color = self.get_next_color()
        if color == 'Next':
            color = self.get_next_color()
        self.forecast_data.append((x_true, y_true, true_label, x, y, label, true_color, color))
    # end def

    def get_next_color(self) -> str:
        colors = ['blue', 'green', 'orange', 'red']
        index = len(self.list_of_colors) % len(colors)
        color = colors[index]
        self.list_of_colors.append(color)
        return(color)
    # end def

    def save(self, path: str) -> None:
        if self.known_data or self.forecast_data:
            plt.figure()

            for x, y, label, color in self.known_data:
                plt.plot(x, y, label = label, color = color)

            for x_true, y_true, true_label, x, y, label, true_color, color in self.forecast_data:
                plt.plot(x_true, y_true, label = true_label, color = true_color)
                plt.plot(x, y, label = label, color = color)

            filepath = os.path.join(BASE_OUTPUT_PATH, path)
            plt.savefig(filepath)
        else:
            raise ValueError('No data to save')
    # end def

    def plot(self) -> None:
        plt.figure()

        for x, y, label, color in self.known_data:
            plt.plot(x, y, label = label, color = color)

        for x_true, y_true, true_label, x, y, label, true_color, color in self.forecast_data:
            plt.plot(x_true, y_true, label = true_label, color = true_color)
            plt.plot(x, y, label = label, color = color)

        plt.title('Известные и прогнозируемые данные до COVID')
        plt.xlabel('Дата')
        plt.ylabel('Стоимость')
        plt.legend()
        plt.show()
    # end def
# end class



# from sktime.utils.plotting import plot_series

# import pandas as pd

    # def plot(self) -> None:
    #     if not self.known_data and not self.forecast_data:
    #         raise ValueError("No data to plot")

    #     plt.figure()

    #     known_series = [(pd.Series(y, index = x), label) for (x, y, label, color) in self.known_data]

    #     forecast_series = []
    #     for x_true, y_true, true_label, x, y, label, true_color, color in self.forecast_data:
    #         forecast_series.append((pd.Series(y_true, index = x_true), true_label, pd.Serise(y, index = x), label))
        
    #     if known_series:
    #         plot_series(*[series for (series, label) in known_series], labels=[label for (series, label) in known_series])
        
    #     for series_true, true_label, series_forecast, label, true_color, color in forecast_series:
    #         plot_series(series_true, series_forecast, labels=[true_label, label], colors=[true_color, color])
        
    #     plt.title('Forecast and Known Data')
    #     plt.xlabel('Time')
    #     plt.ylabel('Value')
    #     plt.show()