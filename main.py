import os

import numpy as np

from frequency_classifier import classify_frequencies
from plot import plot_classification_map, plot_wavelet_map
from reader import read_signals
from wavelet_transformer import compute_wavelet_transform


def main(
        file_path: str,
        omega: int
):
    # TODO: рисовать карты нескольких сигналов в цикле
    # Считывание сигналов
    signals = read_signals(file_path=file_path, delimiter=' ')
    time = signals[:, 0]
    signal_name = 'hk100_1_q0_150_gam0_O_kk100000'
    signal = signals[:, 1]  # Работаем с первым сигналом

    # Вейвлет-преобразование
    time, frequencies, coefficients = compute_wavelet_transform(
        signal=signal,
        time=time,
        omega=omega,
        wavelet='morl'
    )
    # Карта вейвлет-преобразования
    plot_wavelet_map(
        signal_name=signal_name,
        time=time,
        frequencies=frequencies,
        coefficients=coefficients
    )

    # Классификация частот
    classification_map = np.load(f'classification_map_{signal_name}.npy') \
        if os.getenv('LOAD_CLASSIFICATION_MAP', 'False') == 'True' \
        else classify_frequencies(
            signal_name=signal_name,
            coefficients=coefficients,
            frequencies=frequencies,
            time=time,
            omega=omega
        )
    # Построение карты классов
    plot_classification_map(
        signal_name=signal_name,
        classification=classification_map,
        frequencies=frequencies,
        time=time
    )


if __name__ == "__main__":
    os.environ['SAVE_PLOT'] = 'True'
    os.environ['LOAD_CLASSIFICATION_MAP'] = 'True'
    main(
        file_path='data/hk100_1_q0_150_gam0_O_kk100000',
        omega=25
    )
