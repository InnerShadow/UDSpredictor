import os
import pandas as pd
#hui2
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

    def _load_data(self) -> None:
        # Загрузка данных из XLS файла, пропуская первые 5 строк
        self.df = pd.read_excel(self.file_path, header=None, skiprows=5)
    # end def

    def _preprocess_data_frame(self) -> None:
        # Извлечение данных для Date из 1-й колонки (индекс 0) и Cost из 51-й колонки (индекс 50)
        self.df = self.df[[0, 50]]  # Колонки: 0 для Date и 50 для Cost
        self.df.columns = ['Date', 'Cost']  # Переименовываем колонки
        self.df.dropna(inplace=True)  # Удаляем строки с отсутствующими значениями
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')  # Преобразуем даты в формат datetime
        self.df.dropna(subset=['Date'], inplace=True)  # Удаляем строки с некорректными датами
    # end def
    
# end class


class XLSDataLoader(DataLoader):
    pass
# end class

# Функция для объединения данных из нескольких файлов
def load_multiple_years(file_pattern: str, start_year: int, end_year: int, do_scale: bool = False) -> pd.DataFrame:
    combined_df = pd.DataFrame()  # Пустой DataFrame для хранения объединённых данных
    
    for year in range(start_year, end_year + 1):
        file_path = file_pattern.format(year)  # Формируем имя файла, подставляя год
        print(f"Загрузка данных из файла: {file_path}")
        
        loader = XLSDataLoader(file_path, do_scale)  # Создаём экземпляр загрузчика данных
        combined_df = pd.concat([combined_df, loader.df], ignore_index=True)  # Объединяем данные
    # end for
    
    return combined_df
# end def

## Пример использования:
#file_pattern = "{}_day_ru.xls"  # Шаблон названия файлов
#start_year = 2018
#end_year = 2024

## Загрузка и объединение данных за все годы
#full_dataset = load_multiple_years(file_pattern, start_year, end_year, do_scale=True)

## Теперь full_dataset содержит объединённые данные из всех файлов
#print(full_dataset.head())


