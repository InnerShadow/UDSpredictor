import os
import pandas as pd

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod
from datetime import datetime

from Config.config import XLS_FILES

class DataLoader(ABC):
    def __init__(self, do_scale: bool = False) -> None:
        self.df: DataFrame = None
        self._load_data() 
        self._preprocess_data_frame()  
        if do_scale:
            self._do_scale() 
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
        combined_df = pd.DataFrame()  # Пустой DataFrame для хранения объединённых данных
        
        for file_path in XLS_FILES:
            # Загружаем данные из каждого файла Excel, пропуская первые 5 строк
            df = pd.read_excel(file_path, header=None, skiprows=5)
            # Объединяем данные в общий DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        # end for
        
        self.df = combined_df  # Сохраняем объединённый DataFrame как атрибут класса
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

## Пример использования:
# file_pattern = "{}_day_ru.xls"  # Шаблон названия файлов
# start_year = 2018
# end_year = 2024

## Загрузка и объединение данных за все годы через класс DataLoader
# full_dataset = DataLoader(do_scale=False)

## Теперь full_dataset.df содержит объединённые данные из всех файлов
# print(full_dataset.df.head())
