
from numpy import ndarray

from DrawingSystem.DrawingSystem import DrawingSystem

from Config.config import BASE_OUTPUT_PATH

from math import ceil

import os

import matplotlib.pyplot as plt


class CombineDrawer(DrawingSystem):
    def __init__(self) -> None:
        super.__init__()
        self.image_paths : list[str] = []
    #end def
          
    def add_list(self, paths : list[str]) -> None:
        self.image_paths.extend(paths)
    #end def    

    def add_single_path(self, path : str) -> None:
        self.image_paths.append(path)
    # end def

    def combine_images(self, ncols : int):
        nrows = ceil(len(self.image_paths) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize = (5 * ncols, 5 * nrows))
        axs = axs.flatten()

        for i, img_path in enumerate(self.image_paths):
            img = plt.imread(img_path)
            axs[i].imshow(img)
            axs[i].axis('off')
        for j in range(i + 1, nrows * ncols):
            axs[j].axis('off')
        return fig
    #end def

    def save(self, path: str) -> None:
        fig = self.combine_images()
        filepath = os.path.join(BASE_OUTPUT_PATH, path)
        plt.savefig(filepath)
        plt.close(fig)
    # end def

    def plot(self) -> None:
        fig = self.combine_images()
        plt.show()
    # end def
# end class
 