import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from numpy.fft import rfft, irfft, rfftfreq
from numpy.random import uniform


def move_to_axe(arr):
    s = 0
    for one in arr:
        s += one
    mid = s / len(arr)
    arr = np.array(arr) - mid
    return arr


def FH(x):  # ступенчатая функция Хэвисайда
    if x >= 0:
        q = 1
    else:
        q = 0
    return q


def graph_1(name, l1, l2):
    plt.title(name)  # заголовок
    plt.xlabel("x")  # ось абсцисс
    plt.ylabel("y")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(l1, l2)  # построение графика
    plt.show()


def graph_2(title, l1, l2, filename):
    plt.title(title)  # заголовок
    plt.xlabel("x")  # ось абсцисс
    plt.ylabel("y")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(l1, l2)  # построение графика
    plt.savefig(filename)
    figure()


def graph_3(title, time, l1, l2, l3, l4, filename):
    plt.title(title)  # заголовок
    plt.xlabel("x")  # ось абсцисс
    plt.ylabel("y")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(time, l1)
    plt.plot(time, l2)
    plt.plot(time, l3)
    plt.plot(time, l4)
    plt.savefig(filename)
    figure()


##############################################

if __name__ == "__main__":
    time, pos_r, pos_l, pres_up_r, pres_up_l, pres_dn_r, pres_dn_l, trq_r, trq_l = [], [], [], [], [], [], [], [], []
    f = open('1.txt', 'r')
    try:
        # работа с файлом
        text = f.readlines()
        for item in text:
            string = item.split()
            time.append(int(string[0]))
            pos_r.append(int(string[1]))
            pos_l.append(int(string[2]))
            pres_up_r.append(int(string[3]))
            pres_up_l.append(int(string[4]))
            pres_dn_r.append(int(string[5]))
            pres_dn_l.append(int(string[6]))
            trq_r.append(int(string[7]))
            trq_l.append(int(string[8]))
    finally:
        f.close()

    # сдвинем массив pos_r на его срединное значение
    sig = move_to_axe(pos_r)
    # вычисляем преобразование Фурье. Сигнал действительный, поэтому надо использовать rfft, это быстрее, чем fft
    spectrum = rfft(sig)
    N = 289
    FD = 1
    # нарисуем всё это, используя matplotlib
    # Сначала сигнал зашумлённый и тон отдельно
    plt.plot(arange(N) / float(FD), sig)  # по оси времени миллисекунды!
    # plt.plot(arange(N)/float(FD), pure_sig, 'r') # чистый сигнал будет нарисован красным
    plt.xlabel(u'Время, мc')  # для более старых версий питона юникод
    plt.ylabel(u'Позиция, мм')
    plt.title(u'Зашумлённый сигнал и тон 440 Гц')
    plt.grid(True)
    # plt.show()
    plt.savefig('noised.png')

    # когда закроется этот график, откроется следующий
    # Потом спектр
    plt.plot(rfftfreq(N, 1. / FD), np.abs(spectrum) / N)
    # rfftfreq сделает всю работу по преобразованию номеров элементов массива в герцы
    # нас интересует только спектр амплитуд, поэтому используем abs из numpy (действует на массивы поэлементно)
    # делим на число элементов, чтобы амплитуды были в мм, а не в суммах Фурье. Проверить просто — постоянные составляющие должны совпадать в сгенерированном сигнале и в спектре
    plt.xlabel(u'Частота, Гц')
    plt.ylabel(u'Позиция, мм')
    plt.title(u'Спектр')
    plt.grid(True)
    # plt.show()
    plt.savefig('spectre.png')

    # Аппроксимация мнк
    t = np.polyfit(time, pos_r, 8)
    f = np.poly1d(t)
    plt.plot(time, f(time))
    plt.xlabel(u'Время, мc')
    plt.ylabel(u'Позиция, мм')
    plt.title(u'Аппроксимация')
    plt.grid(True)
    # plt.show()
    plt.savefig('appr.png')
