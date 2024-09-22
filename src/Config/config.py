
import os
from torch import optim

BASE_OUTPUT_PATH = os.path.join('Data', 'Output')

XLS_FILES = (
    os.path.join('Data', '2018_day_ru.xls'),
    os.path.join('Data', '2019_day_ru.xls'),
    os.path.join('Data', '2020_day_ru.xls'),
    os.path.join('Data', '2021_day_ru.xls'),
    os.path.join('Data', '2022_day_ru.xls'),
    os.path.join('Data', '2023_day_ru.xls'),
    os.path.join('Data', '2024_day_ru.xls')
)

OPTIMAZER_MAP = {
    'rmsprop' : optim.RMSprop,
    'adam' : optim.AdamW,
    'nadam' : optim.Adam,
    'adamw' : optim.NAdam
}
