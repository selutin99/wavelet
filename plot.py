import os

import matplotlib.image
import numpy as np
from matplotlib import pyplot as plt


def plot_wavelet_map(
        signal_name: str,
        coefficients: np.ndarray,
        frequencies: np.ndarray,
):
    """
    Отображает карту вейвлет-преобразования для заданного сигнала

    :param signal_name: Название сигнала
    :param coefficients: Коэффициенты вейвлет-преобразования
    :param frequencies: Частоты, соответствующие коэффициентам
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.abs(coefficients),
        extent=(0, 1, frequencies[-1], frequencies[0]),
        cmap='jet',
        aspect='auto',
        interpolation='nearest'
    )
    plt.colorbar(label='Модуль коэффициентов')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title(f'Вейвлет-преобразование сигнала: ({signal_name})')

    __save_or_show_plot(signal_name)


def plot_frequency_map(
        signal_name: str,
        classification: np.ndarray,
        frequencies: np.ndarray,
        time: np.ndarray
):
    """
    Визуализирует карту частотных классов на графике, где по оси X - время,
    а по оси Y - частота. Отображаются различные классы частот, назначенные
    для каждой временной точки и частоты

    :param signal_name: Название сигнала
    :param classification: Двумерный массив, содержащий классификацию частот для каждой точки времени
    :param frequencies: Массив частот, соответствующих шкале частот
    :param time: Массив времени, соответствующий временным точкам
    """
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        classification,
        aspect='auto',
        cmap='plasma',
        extent=(time[0], time[-1], frequencies[0], frequencies[-1])
    )

    # Добавление цветовой шкалы
    plt.colorbar(im, label='Классификация')

    # Подписи осей
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title(f'Карта сигнала {signal_name}')

    __add_classification_legend(im)
    __save_or_show_plot(signal_name)


def __add_classification_legend(im: matplotlib.image.AxesImage):
    """
    Добавляет легенду на частотную карту

    :param im: Изображение для отрисовки легенды
    """
    legend_labels = {
        1: 'Гармонические колебания f0',
        2: 'Колебания на частоте f0/2',
        3: 'Колебания на частоте f0/3',
        4: 'Колебания на независимой частоте',
        5: 'Затухающие колебания',
        6: 'Бифуркации',
        7: 'Хаос'
    }

    # Создание легенды
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=im.cmap(im.norm(i)), markersize=10)
        for i in legend_labels.keys()
    ]
    plt.legend(
        handles,
        legend_labels.values(),
        title='Режимы колебаний',
        loc='upper left',
        bbox_to_anchor=(1.2, 1.0)
    )


def __save_or_show_plot(signal_name: str):
    """
    Сохраняет график в файл или отображает его

    :param signal_name: Название сигнала
    """
    if os.getenv('SAVE_PLOT', 'False') == 'True':
        plt.savefig(f'maps/{signal_name}.png', bbox_inches='tight')
    else:
        plt.show()
