from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

from pandas import DataFrame

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import os


class AutocorrelationDrawer(DrawingSystem):

    def __init__(self) -> None:
        super().__init__()
        self.lags: int = 40
    #end def

    def set_data(self, data : DataFrame) -> None:
        self.list_of_data.append(data)
    # end def

    def set_lags(self, lags: int) -> None:
        self.lags = lags
    # end def

    def save(self, path: str) -> None:
        if self.list_of_data:
            fig, axs = plt.subplots(2, 1, figsize = (10, 12))

            plot_acf(self.list_of_data[-1], lags = self.lags, ax = axs[0])
            axs[0].set_title('Autocorrelation Function')

            plot_pacf(self.list_of_data[-1], lags = self.lags, ax = axs[1])
            axs[1].set_title('Partial Autocorrelation Function')
            
            filepath = os.path.join(BASE_OUTPUT_PATH, path)
            plt.savefig(filepath)
        else:
            raise ValueError('No data to save')
    # end def

    def plot(self) -> None:
        fig, axs = plt.subplots(2, 1, figsize = (10, 12))
            
        plot_acf(self.list_of_data[-1], lags = self.lags, ax = axs[0])
        axs[0].set_title('Autocorrelation Function')

        plot_pacf(self.list_of_data[-1], lags = self.lags, ax = axs[1])
        axs[1].set_title('Partial Autocorrelation Function')
        
        plt.tight_layout()
        plt.show()
    # end def

