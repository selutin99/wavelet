import numpy as np
import pywt


def compute_wavelet_transform(
        signal: np.ndarray,
        time: np.ndarray,
        omega: int,
        wavelet: str = 'morl'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Выполняет вейвлет-преобразование сигнала с использованием заданного вейвлета

    :param signal: Одномерный массив, представляющий сигнал для анализа
    :param time: Одномерный массив, представляющий время распространения сигнала
    :param omega: Частота возбуждения сигнала
    :param wavelet: Название вейвлета, используемого для преобразования. По умолчанию используется Морле

    :return: кортеж времени, коэффициентов и частот
        - time: одномерный массив времени
        - frequencies: одномерный массив частот, соответствующих каждому масштабу
        - coefficients: двумерный массив коэффициентов вейвлет-преобразования, где строки - масштабы, а столбцы — время
    """
    dt_w = time[1] - time[0]

    # Параметры преобразования
    min_omega = 0.5
    max_omega = omega + 0.3

    min_f = min_omega / (2 * np.pi)  # Минимальная частота
    max_f = max_omega / (2 * np.pi)  # Максимальная частота

    lss = 0.05
    log_scale_step = lss
    lss_factor = 2 ** log_scale_step

    # Определение масштабов
    min_scale = 1 / (max_f * dt_w)
    max_scale = 1 / (min_f * dt_w)

    scales = [min_scale]
    while scales[-1] < max_scale:
        scales.append(scales[-1] * lss_factor)

    scales = np.array(scales)

    # Выполнение непрерывного вейвлет-преобразования
    wvl_coeffs, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=dt_w)
    abs_c = np.abs(wvl_coeffs)

    # Конвертация частот в радианы
    pffreqs = frequencies * (2 * np.pi)

    # Подготовка данных для отображения
    time = time[::1]
    coefficients = abs_c[:, ::1]

    return time, pffreqs, coefficients
