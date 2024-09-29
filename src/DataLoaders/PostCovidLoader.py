
from Containers.DataLoader import DataLoader

from Config.config import XLS_FILES

class PostCovidLoader(XLSDataLoader):
    def __init__(self, file_path: str, do_scale: bool = False) -> None:
        super().__init__(file_path, do_scale)
    
    def _preprocess_data_frame(self) -> None:
        super()._preprocess_data_frame()  # Предобработка базовых данных
        covid_end = datetime(2021, 6, 1)  # Примерный конец активной фазы пандемии COVID
        exponential_start = datetime(2014, 12, 1)  # Примерный период экспоненциального роста рубля
        
        # Фильтруем данные после окончания активной фазы COVID и исключаем период экспоненциального роста
        self.df = self.df[(self.df['Date'] > covid_end) & (self.df['Date'] < exponential_start)]
    # end def
# end class
