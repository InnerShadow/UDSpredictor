
from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

import matplotlib.pyplot as plt

import os

import scipy.stats as stats

class ResidueDrawer(DrawingSystem):
    def __init__(self) -> None:
        super().__init__()
        self.residuals : ndarray = [] #ndarray()
    # end def

    def add_residue(self, 
                    x_true : ndarray, 
                    x_predicted : ndarray, 
                    cmap : str = 'viridis') -> None:
        residue = x_true - x_predicted
        self.residuals.append((x_true, x_predicted, cmap))
    # end def

    def save(self, path: str) -> None:
        if self.residuals:
            fig, axs = plt.subplots(3, 1, figsize = (10, 12))

            for x_true, residue, cmap in self.residuals:
                #axs[0] = plot(x_true, residue, marker = ... line cmap )
                scatter = axs[0].scatter(x_true, residue, color = 'blue' , cmap = cmap)
                axs[0].axhline(y = 0, color = 'red', linestyle = '--')
                axs[0].set_title('Residuals')
                axs[0].set_xlabel('True Values')
                axs[0].set_ylabel('Residuals')

                stats.probplot(residue, dist = 'norm', plot = axs[1])
                axs[1].set_title('Q-Q Plot of the Residuals')
            
                #bins???
                axs[2].hist(residue, bins = 30, color = 'blue', alpha = 0.7, edgecolor = 'black')
                axs[2].set_title('Histogram of the Residuals')
                axs[2].set_xlabel('Residuals')
                axs[2].set_ylabel('Frequency')

            filepath = os.path.join(BASE_OUTPUT_PATH, path)
            plt.savefig(filepath)
        else:
            raise ValueError('No data to save')
    # end def

    # Y OR X
    def plot(self) -> None:
        fig, axs = plt.subplots(3, 1, figsize = (10, 12))

        for x_true, residue, cmap in self.residuals:
            scatter = axs[0].scatter(x_true, residue, color = 'blue' , cmap = cmap)
            axs[0].axhline(y = 0, color = 'red', linestyle = '--')
            axs[0].set_title('Residuals')
            axs[0].set_xlabel('True Values')
            axs[0].set_ylabel('Residuals')

            stats.probplot(residue, dist = 'norm', plot = axs[1])
            axs[1].set_title('Q-Q Plot of the Residuals')
            
            axs[2].hist(residue, bins = 30, color = 'blue', alpha = 0.7, edgecolor = 'black')
            axs[2].set_title('Histogram of the Residuals')
            axs[2].set_xlabel('Residuals')
            axs[2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()
    # end def
# end class
