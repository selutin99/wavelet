import numpy as np
import logging

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Константы для классификации
CLASSIFICATION = {
    'HARMONIC': 1,
    'OMEGA_DIV_3': 2,
    'OMEGA_DIV_2': 3,
    'INDEPENDENT': 4,
    'DAMPED': 5,
    'BIFURCATION': 6,
    'CHAOS': 7,
}


def classify_frequencies(
        signal_name: str,
        coefficients: np.ndarray,
        frequencies: np.ndarray,
        time: np.ndarray,
        omega: int
) -> np.ndarray:
    """
    Классифицирует частоты на основе вейвлет-коэффициентов и возвращает карту классификации

    :param signal_name: Наименование сигнала
    :param coefficients: Вейвлет-коэффициенты
    :param frequencies: Частоты
    :param time: Массив времени распространения сигнала
    :param omega: Базовая частота
    :return: ndarray Карта классов частот
    """
    num_scales, num_times = coefficients.shape
    classification_map = np.zeros((num_scales, num_times), dtype=int)

    # Расчет базовых параметров
    domega = (2 * np.pi) / (num_times * (time[1] - time[0]))
    decay_threshold = 0.1  # Порог для затухающих колебаний
    chaos_threshold = 0.2  # Порог для хаотических сигналов

    for scale_idx, freq in enumerate(frequencies):
        for time_idx in range(num_times):
            classification_map[scale_idx, time_idx] = __classify_point(
                data=coefficients[scale_idx, :],
                freq=freq,
                omega=omega,
                time_idx=time_idx,
                domega=domega,
                decay_threshold=decay_threshold,
                chaos_threshold=chaos_threshold
            )
        logger.info(f'Классификация завершена для шкалы {scale_idx + 1} из {num_scales}')
    np.save(f'classification_map_{signal_name}.npy', classification_map)
    return classification_map


def __classify_point(
        data: np.ndarray,
        freq: float,
        omega: float,
        time_idx: int,
        domega: float,
        decay_threshold: float,
        chaos_threshold: float
) -> int:
    """
    Классифицирует конкретную точку данных.

    :param data: Данные для текущей шкалы
    :param freq: Частота
    :param omega: Базовая частота
    :param time_idx: Индекс времени
    :param domega: Допустимое отклонение частоты
    :param decay_threshold: Порог для затухающих колебаний
    :param chaos_threshold: Порог для хаотических сигналов
    :return: Класс частоты
    """
    if freq == omega:
        return CLASSIFICATION['HARMONIC']
    elif np.isclose(freq, omega / 3, atol=domega):
        return CLASSIFICATION['OMEGA_DIV_3']
    elif np.isclose(freq, omega / 2, atol=domega):
        return CLASSIFICATION['OMEGA_DIV_2']
    elif np.std(data) > chaos_threshold:
        return CLASSIFICATION['CHAOS']
    elif np.abs(data[time_idx]) < decay_threshold:
        return CLASSIFICATION['DAMPED']
    elif np.any(np.diff(data) < 0) and np.any(np.diff(data) > 0):
        return CLASSIFICATION['BIFURCATION']
    else:
        return CLASSIFICATION['INDEPENDENT']
