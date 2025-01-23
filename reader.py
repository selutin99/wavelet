import numpy as np


def read_signals(file_path: str, delimiter: str = ',') -> np.ndarray:
    """
    Считывает сигналы из файла в формате CSV и возвращает их в виде pandas DataFrame

    :param file_path: Путь к файлу CSV, где первый столбец — временная шкала, а последующие столбцы — амплитуды сигналов
    :param delimiter: Разделитель в файле CSV. По умолчанию запятая

    :return: DataFrame, где строки соответствуют временным точкам, а столбцы — сигналам (включая временную шкалу)
    """
    return np.loadtxt(file_path, delimiter=delimiter)
