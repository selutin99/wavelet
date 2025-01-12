import numpy as np
import pandas
import pywt


def compute_wavelet_transform(
        signal: pandas.Series,
        scales: np.ndarray,
        wavelet: str = 'cmor'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Выполняет вейвлет-преобразование сигнала с использованием заданного вейвлета

    :param signal: Одномерный массив, представляющий сигнал для анализа
    :param scales: Массив масштабов для выполнения преобразования. Масштабы определяют временно-частотное разрешение
    :param wavelet: Название вейвлета, используемого для преобразования. По умолчанию используется комплексный Морле

    :return: кортеж коэффициентов и частот
        - coefficients: двумерный массив коэффициентов вейвлет-преобразования, где строки - масштабы, а столбцы — время
        - frequencies: одномерный массив частот, соответствующих каждому масштабу
    """
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    return coefficients, frequencies
