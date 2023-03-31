from numpy import sin, cos, sqrt, arctan
import numpy as np
import matplotlib.pyplot as plt

G = 9.8  # ускорение свободного падения, м/с^2
L1 = 0.59  # длина бедра, м
L2 = 0.53  # длина голени, м
L3 = 0.75  # длина корпуса, м
M1 = 9.9  # масса бедра, кг
M2 = 7.2  # масса голени, кг
M3 = 25.  # масса корпуса, кг
h = 1.11  # высота точки подвеса
s = 0.2  # опорный сдвиг
L_step = 0.5  # длина шага
Ampl = 0.2  # амплитуда синусоиды движения переносной ноги
T = 1.1  # период двойного шага
omega = 2 * 3.14 / T  # угловая скорость
dt = 0.05  # изменение времени
t = np.arange(0, 7, dt)


def find_knee(x1, y1, x2, y2):
    # ищем точки пересечения двух окружностей 
    # с центом в тазу и в пятке с радиусами L1, L2
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # нет пересечения
    if d > L1 + L2:
        # print("case 1")
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    # есть пересечения
    a = (L2 ** 2 - L1 ** 2 + d ** 2) / (2 * d)
    h_ = sqrt(L2 ** 2 - a ** 2)
    # середина
    xc = x1 + a * (x2 - x1) / d
    yc = y1 + a * (y2 - y1) / d
    # первая точка пересечения
    x3 = xc - h_ * (y2 - y1) / d
    y3 = yc + h_ * (x2 - x1) / d
    # вторая точка пересечения
    x4 = xc + h_ * (y2 - y1) / d
    y4 = yc - h_ * (x2 - x1) / d
    # print("3-1 dst=", sqrt((x3-x1)**2+(y3-y1)**2), "3-2 dst=", sqrt((x3-x2)**2+(y3-y2)**2))
    # если такая точка одна
    if x3 == x4 and y3 == y4:
        # print("case 2")
        return [x3, y3]
    # проверка того, что угол сгиба колена <= 180
    # в нашем случае достаточно, чтобы xi был больше xj (идем вправо)
    if x3 > x4:
        # print("case 3")
        return [x3, y3]
    else:
        # print("case 4")
        return [x4, y4]


# построение графика всех параметров от времени на одном графике
def graph_draw(title, x0, y0, x1_1, y1_1, x1_2, y1_2, x3, y3, filename, legend=['Голова', 'Колено1', 'Колено2', 'Таз']):
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("dots")
    plt.grid()
    plt.plot(x0, y0)
    plt.plot(x1_1, y1_1)
    plt.plot(x1_2, y1_2)
    plt.plot(x3, y3)
    plt.legend(legend)
    plt.savefig(filename)
    plt.figure()


if __name__ == "__main__":
    print("start")
    # угол наклона корпуса 2-периодичный зададим вручную
    psi = np.radians(-4.3 + 2.7 * sin(2 * omega * t) - 1.5 * cos(2 * omega * t))
    # таз
    x0 = np.linspace(0, 7, len(t))
    y0 = np.ones(len(t)) * h
    # пятка 1
    x1_2 = t + L_step / np.pi * (- np.sin(omega * t))
    y1_2 = L_step * 0.1 * (1 - np.cos(omega * t))

    # колено 1
    print("finding coordinates of knee1...")
    x1_1 = [find_knee(x0[i], y0[i], x1_2[i], y1_2[i])[0] for i in range(len(t))]
    y1_1 = [find_knee(x0[i], y0[i], x1_2[i], y1_2[i])[1] for i in range(len(t))]
    """
    # пятка 2
    x2_2 = L_step / np.pi / 2 * (2*t - np.sin(omega*t - np.pi)) - 1.55
    y2_2 = L_step * 0.1 *(1 - np.cos(omega*t - np.pi))
    # колено 2
    print("finding coordinates of knee2...")
    x2_1 = [find_knee(x0[i], y0[i], x2_2[i], y2_2[i])[0] for i in range(len(t))]
    y2_1 = [find_knee(x0[i], y0[i], x2_2[i], y2_2[i])[1] for i in range(len(t))]
    """
    # голова
    x3 = x0 - L3 * sin(psi)
    y3 = y0 + L3 * cos(psi)

    graph_draw("Траектории точек экзоскелета", x0, y0, x1_1, y1_1, x1_2, y1_2, x3, y3, "tracks.png",
               legend=['таз', 'колено', 'пятка', 'голова'])
    print("end")
