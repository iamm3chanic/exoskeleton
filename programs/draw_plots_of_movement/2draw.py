# 2draw.py
import matplotlib.pyplot as plt
import numpy as np
from pylab import *


def graphs_t(pos, pres_up, pres_dn, trq):
    x = np.linspace(0, 10, 50)
    y1 = x
    # Квадратичная зависимость
    y2 = [i ** 2 for i in x]
    # Построение графика
    plt.title("Зависимости: y1 = x, y2 = x^2")  # заголовок
    plt.xlabel("x")  # ось абсцисс
    plt.ylabel("y1, y2")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y1, x, y2)  # построение графика
    plt.show()


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


def graph_dot(title, l1, l2, filename):
    plt.title(title)  # заголовок
    plt.xlabel("x")  # ось абсцисс
    plt.ylabel("y")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.scatter(l1, l2)  # построение точечногографика
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


if __name__ == "__main__":
    time, pos_r, pos_l, pres_up_r, pres_up_l, pres_dn_r, pres_dn_l, trq_r, trq_l = [
    ], [], [], [], [], [], [], [], []
    f = open('1_1.txt', 'r')
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
            # str(f.readline(i+1)) #.split()
            # i+=1
    finally:
        f.close()

    graph_2("Правый, pos от времени", time, pos_r, "pos__time_r.png")

    graph_2(
        "Правый, pres_up от времени",
        time,
        pres_up_r,
        "pres_up__time_r.png")

    graph_2(
        "Правый, pres_dn от времени",
        time,
        pres_dn_r,
        "pres_dn__time_r.png")

    graph_2("Правый, trq от времени", time, trq_r, "trq__time_r.png")

    graph_2("Левый, pres_up от времени", time,
            pres_up_l, "pres_up__time_l.png")

    graph_2("Левый, pres_dn от времени", time,
            pres_dn_l, "pres_dn__time_l.png")

    # с нужной выборкой!
    pos_r_short, pdn_r_short = pos_r[0:150], pres_dn_r[0:150]
    graph_dot(
        "Правый, pres_dn от pos",
        pos_r_short,
        pdn_r_short,
        "pres_dn__pos_r.png")
    graph_dot("Левый, pres_dn от pos", pos_l, pres_dn_l, "pres_dn__pos_l.png")

    graph_3("Правый: pos, pres_up, pres_dn, trq от времени",
            time, pos_r, pres_up_r, pres_dn_r, trq_r, "all.png")


    figure()
