
from DataLoaders.DataLoader import DataLoader

from Config.config import XLS_FILES

class PreCovidLoader(XLSDataLoader):
    def __init__(self, file_path: str, do_scale: bool = False) -> None:
        super().__init__(file_path, do_scale)
    
    def _preprocess_data_frame(self) -> None:
        super()._preprocess_data_frame()  # Предобработка базовых данных
        covid_start = datetime(2020, 3, 1)  # Начало пандемии COVID
        exponential_start = datetime(2014, 12, 1)  # Примерный период экспоненциального роста рубля
        
        # Фильтруем данные до начала COVID и исключаем период экспоненциального роста рубля
        self.df = self.df[(self.df['Date'] < covid_start) & (self.df['Date'] < exponential_start)]
    # end def
# end class
