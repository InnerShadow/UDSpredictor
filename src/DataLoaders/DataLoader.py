import os
import pandas as pd
#hui
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Config.config import XLS_FILES

from Config.config import USD_COLUMN_VALUE

class DataLoader():
    def __init__(self, do_scale : str) -> None:
        self.df: DataFrame = None
        self._load_data() 
        self._preprocess_data_frame()  


        # TODO: Обработать 2 типа скейлов по значению do_scale, которые может принимать занчения "min" и "std"
       
    # end def

    def get_time_prediod(self, perido : str) -> DataFrame:
        if perido == 'pre':
            self._get_data_pre_coiv()
        elif perido == 'post':
            pass
        else:
            raise ValueError('Bad period.')
        # end if
    # end def

    def _get_data_pre_coiv(self) -> DataFrame:
        pass

    def _load_data(self) -> None:                     
        pass
    # end def

    def _preprocess_data_frame(self) -> None:         
        pass
    # end def

    def __len__(self) -> int:
        return len(self.df)
    # end def

    def __getitem__(self, index: int) -> float:
        return self.df.iloc[index]['Cost']
    # end def

    def _do_min_max_scale(self) -> None:
        scaler = MinMaxScaler()
        self.df['Cost'] = scaler.fit_transform(self.df[['Cost']])
    # end def
    
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


