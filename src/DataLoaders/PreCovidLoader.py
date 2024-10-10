
from DataLoaders.DataLoader import DataLoader

from Config.config import XLS_FILES
from datetime import datetime

class PreCovidLoader(DataLoader):
    def __init__(self, do_scale: bool = False) -> None:
        super().__init__(do_scale)
    
    def _preprocess_data_frame(self) -> None:
        super()._preprocess_data_frame()  # Предобработка базовых данных
        covid_start = datetime(2020, 3, 1)  # Начало пандемии COVID
        exponential_start = datetime(2022, 2, 21)  # Примерный период экспоненциального роста рубля
        
        # Фильтруем данные до начала COVID и исключаем период экспоненциального роста рубля
        self.df = self.df[(self.df['Date'] < covid_start) & (self.df['Date'] < exponential_start)]
    # end def
# end class
