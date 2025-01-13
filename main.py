import numpy as np

from frequency_classifier import classify_frequencies
from plot import plot_frequency_map, plot_wavelet_map
from reader import read_signals
from wavelet_transformer import compute_wavelet_transform


def main(
        file_path: str,
        excitation_frequency: float,
        scales=np.arange(1, 128)
):
    # TODO: рисовать карты нескольких сигналов в цикле
    # Считывание сигналов
    signals = read_signals(file_path=file_path, delimiter=' ')
    time = signals.index
    signal_name = signals.columns[1]
    signal = signals.iloc[:, 1]  # Работаем с первым сигналом

    # Вейвлет-преобразование
    coefficients, frequencies = compute_wavelet_transform(
        signal=signal,
        scales=scales,
        wavelet='morl'
    )

    # Карта вейвлет-преобразования
    plot_wavelet_map(signal_name, coefficients, frequencies)

    # Анализ частот
    classification = classify_frequencies(coefficients, frequencies, excitation_frequency)

    # Построение частотной карты
    plot_frequency_map(signal_name, classification, frequencies, time)


if __name__ == "__main__":
    main(
        file_path='data/signal.csv',
        excitation_frequency=0.5
    )
