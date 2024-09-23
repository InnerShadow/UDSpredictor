import os
import pandas as pd

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod

class DataLoader(ABC):
    def __init__(self, file_path: str, do_scale: bool = False) -> None:
        self.file_path = file_path
        self.df: DataFrame = None
        self._load_data() 
        self._preprocess_data_frame()  
        if do_scale:
            self._do_scale() 
    # end def

    @abstractmethod
    def _load_data(self) -> None:                     
        raise NotImplementedError()
    # end def

    @abstractmethod
    def _preprocess_data_frame(self) -> None:         
        raise NotImplementedError()
    # end def

    def __len__(self) -> int:
        return len(self.df)
    # end def

    def __getitem__(self, index: int) -> float:
        return self.df.iloc[index]['Cost']
    # end def

    def _do_scale(self) -> None:
        scaler = MinMaxScaler()
        self.df['Cost'] = scaler.fit_transform(self.df[['Cost']])
    # end def
    
# end class

# Реализация конкретного загрузчика данных
def _preprocess_data_frame(self) -> None:                              #Предобработка данных: выбор колонок с датой и курсом USD.
                                                                       # Ищем столбцы с датой и курсом доллара США
        usd_column_name = 'Доллар США (USD)'
        if 'Дата' not in self.df.columns or usd_column_name not in self.df.columns:
            raise ValueError("Датасет должен содержать колонки 'Дата' и 'Доллар США (USD)'.")
        
                                                                            # Выбираем только колонки с датой и курсом USD
        self.df = self.df[['Дата', usd_column_name]]
        self.df.columns = ['Date', 'Cost']                                            # Переименовываем для единообразия
        
                                                                                # Преобразуем 'Date' в datetime формат, если это не так
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
    # end def
# end class
