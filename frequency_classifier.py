import numpy as np


def classify_frequencies(
        coefficients: np.ndarray,
        frequencies: np.ndarray,
        excitation_frequency: float
) -> np.ndarray:
    """
    Классифицирует частоты на основе их значений и связи с возбуждающей частотой.

    :param coefficients: Коэффициенты вейвлет-преобразования, характеризующие амплитуды частот.
    :param frequencies: Частоты, соответствующие масштабам вейвлет-преобразования.
    :param excitation_frequency: Основная частота возбуждения (f0).

    :return: Массив классификации частот для каждого времени:
        - 1: Частота возбуждения (гармоническая)
        - 2: Частота, равная f0 / 2
        - 3: Частота, равная f0 / 3
        - 4: Независимая частота
        - 5: Затухающие колебания
        - 6: Бифуркации
        - 7: Хаос
    """
    eps = 1e-3
    classification = np.zeros_like(coefficients, dtype=np.int8)

    for t in range(coefficients.shape[1]):
        for f_idx, freq in enumerate(frequencies):
            if np.abs(freq - excitation_frequency) < eps:
                classification[f_idx, t] = 1  # Частота возбуждения
            elif np.abs(freq - excitation_frequency / 2) < eps:
                classification[f_idx, t] = 2  # Частота f0/2
            elif np.abs(freq - excitation_frequency / 3) < eps:
                classification[f_idx, t] = 3  # Частота f0/3
            elif (np.abs(freq - excitation_frequency / 2) > eps and
                  np.abs(freq - excitation_frequency / 3) > eps and
                  coefficients[f_idx, t] < 0.1):
                classification[f_idx, t] = 4  # Независимая частота
            elif (np.abs(freq - excitation_frequency / 3) > eps and
                  coefficients[f_idx, t] < 0.1):
                classification[f_idx, t] = 5  # Затухающие колебания
            elif (np.abs(freq - excitation_frequency) > eps and
                  np.abs(freq - excitation_frequency / 2) > eps
                  and np.abs(freq - excitation_frequency / 3) > eps):
                classification[f_idx, t] = 6  # Бифуркации
            else:
                classification[f_idx, t] = 7  # Хаос

    return classification
