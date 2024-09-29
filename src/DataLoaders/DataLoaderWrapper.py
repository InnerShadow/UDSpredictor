
from pandas import DataFrame

from DataLoaders.PostCovidLoader import PostCovidLoader
from DataLoaders.PreCovidLoader import PreCovidLoader

class DataLoaderWrapper:
    def __init__(self, period: str, file_path: str, do_scale: bool = False) -> None:
        if period == 'pre':
            self.dataLoader = PreCovidLoader(file_path, do_scale=do_scale)
        elif period == 'post':
            self.dataLoader = PostCovidLoader(file_path, do_scale=do_scale)
        else:
            raise ValueError('Bad period!')
        # end if
    # end def

    def get_data(self) -> DataFrame:
        return self.dataLoader.df
    # end def
# end class

