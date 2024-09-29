
from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

import matplotlib.pyplot as plt

import os

class MultiChartDrawer(DrawingSystem):

    def __init__(self) -> None:
        super().__init__()
        self.charts : ndarray = ndarray()
    # end def

    def add_new_chart(self, 
                      x : ndarray, 
                      y : ndarray, 
                      label : str,
                      color : str = 'Next') -> None:
        if color == 'Next':
            color = self.get_new_color()
        self.charts.append((x, y, label, color))
    # end def

    def get_new_color(self) -> str:
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        index = len(self.list_of_colors) % len(colors)
        color = colors[index]
        self.list_of_colors.append(color)
        return color
    # end def   

    def save(self, path: str) -> None:
        if self.charts:
            filepath = os.path.join(BASE_OUTPUT_PATH, path)
            plt.savefig(filepath)
        else:
            raise ValueError('No charts to save')
    # end def

    def plot(self) -> None:
        plt.figure()

        for x, y, label, color in self.charts:
            plt.plot(x, y, label = label, color = color)
    
        plt.legend()
        plt.show()
    # end def
# end class
