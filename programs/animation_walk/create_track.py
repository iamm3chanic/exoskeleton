from numpy import sin, cos, sqrt, arctan
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

G = 9.8  # ускорение свободного падения, м/с^2
L1 = 0.59  # длина бедра, м
L2 = 0.53  # длина голени, м
L3 = 0.75  # длина корпуса, м
M1 = 9.9  # масса бедра, кг
M2 = 7.2  # масса голени, кг
M3 = 25.  # масса корпуса, кг
h = 1.09  # высота точки подвеса
s = 0.2  # опорный сдвиг
L_step = 0.5  # длина шага
Ampl = 0.05  # амплитуда синусоиды движения переносной ноги
T = 1.1  # период двойного шага
omega = 2 * 3.14 / T  # угловая скорость
dt = 0.05  # изменение времени
t = np.arange(0, 7.7, dt)


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


def find_angles(x0, y0, x1_1, y1_1, x1_2, y1_2, x2_1, y2_1, x2_2, y2_2):
    alpha1, beta1, alpha2, beta2 = [], [], [], []
    for i in range(len(x0)):
        # бедро1 - вертикаль
        alpha1.append(np.arctan((x1_1[i] - x0[i]) / (y1_1[i] - y0[i])))
        # голень1 - вертикаль
        beta1.append(np.arctan((x1_2[i] - x1_1[i]) / (y1_2[i] - y1_1[i])))
        # бедро2 - вертикаль
        alpha2.append(np.arctan((x2_1[i] - x0[i]) / (y2_1[i] - y0[i])))
        # голень2 - вертикаль
        beta2.append(np.arctan((x2_2[i] - x2_1[i]) / (y2_2[i] - y2_1[i])))
    return alpha1, beta1, alpha2, beta2


def count_energy(alpha1, beta1, alpha2, beta2, psi, q1, q2, u1, u2):
    # потенциальная энергия: сначала прибавим энергию от корпуса
    energy1 = M3 * G * y3 * cos(psi)
    # прибавим энергию от первого бедра
    energy2 = -M1 * G * L1 * cos(alpha1)
    # прибавим энергию от первой голени
    energy3 = -M2 * G * (2 * L1 * cos(alpha1) + L2 * cos(beta1))
    # прибавим энергию от второго бедра
    energy4 = -M1 * G * L1 * cos(alpha2)
    # прибавим энергию от второй голени
    energy5 = -M2 * G * (2 * L1 * cos(alpha2) + L2 * cos(beta2))
    # полная потенциальная энергия относительно таза
    energy = energy1 + energy2 + energy3 + energy4 + energy5

    # кинетическая энергия (обобщенный силы): сначала прибавим Q от корпуса
    Q1 = q1 + q2 + M3 * G * y3 * sin(psi)
    # прибавим Q от первого бедра
    Q2 = q1 - u1 - M1 * G * L1 * sin(alpha1) - 2 * M2 * G * L1 * sin(alpha1)
    # прибавим Q от первой голени
    Q3 = q1 - M2 * G * L2 * sin(beta1)
    # прибавим Q от второго бедра
    Q4 = q2 - u2 - M1 * G * L1 * sin(alpha2) - 2 * M2 * G * L1 * sin(alpha2)
    # прибавим Q от второй голени
    Q5 = q2 - M2 * G * L2 * sin(beta2)
    # полная кинетическая энергия (обобщенный силы)
    Q = Q1 + Q2 + Q3 + Q4 + Q5

    return energy, Q


if __name__ == "__main__":
    print("start")
    # угол наклона корпуса 2-периодичный зададим вручную
    psi = np.radians(-4.3 + 2.7 * sin(2 * omega * t) - 1.5 * cos(2 * omega * t))
    # таз
    x0 = np.linspace(0, 7, len(t))
    y0 = np.ones(len(t)) * h
    # пятка 1
    # x1_2 = t + L_step / np.pi * (- np.sin(omega * t))
    # y1_2 = Ampl * (1 - np.cos(omega * t))
    period = np.arange(0, 1.1, dt)
    half = np.arange(0, 0.55, dt)
    x1_2p = np.copy(period)  # period - L_step /2 * np.sin(omega * period)
    x1_2p[:len(x1_2p) // 2] = half * 2 - L_step / np.pi * np.sin(omega * half)
    x1_2p[len(x1_2p) // 2:] = x1_2p[len(x1_2p) // 2 - 1]
    x1_2 = np.copy(x0)
    for i in range(7):
        x1_2[i * 22: (i + 1) * 22] = x1_2p + np.ones(22) * (x0[i * 22] - L_step / 2)  # -np.ones(22)*0.2

    y1_2p = np.zeros(len(period))
    y1_2p[:] = Ampl * (1 - np.cos(omega * period * 2))
    y1_2p[len(x1_2p) // 2:] = 0
    y1_2 = np.zeros(len(t))
    for i in range(7):
        y1_2[i * 22: (i + 1) * 22] = y1_2p

    # колено 1
    print("finding coordinates of knee1...")
    x1_1 = [find_knee(x0[i], y0[i], x1_2[i], y1_2[i])[0] for i in range(len(t))]
    y1_1 = [find_knee(x0[i], y0[i], x1_2[i], y1_2[i])[1] for i in range(len(t))]
    # пятка 2
    # x2_2 = t + L_step / np.pi * (- np.sin(omega * t - np.pi))
    x2_2p = np.copy(period) #-L_step / 2 * np.sin(omega * period - np.pi)

    x2_2p[len(x1_2p) // 2:] = half * 2 - L_step / np.pi * np.sin(omega * half) #+np.pi
    x2_2p[:len(x2_2p) // 2] = x2_2p[0]
    x2_2 = np.copy(x0)
    for i in range(7):
        x2_2[i * 22: (i + 1) * 22] = x2_2p + np.ones(22) * (x0[i * 22] + L_step/2)
    #y2_2 = Ampl * (1 - np.cos(omega * t - np.pi))
    y2_2p = np.zeros(len(period))
    y2_2p[:] = Ampl * (1 - np.cos(omega * period * 2))
    y2_2p[:len(x1_2p) // 2] = 0
    y2_2 = np.zeros(len(t))
    for i in range(7):
        y2_2[i * 22: (i + 1) * 22] = y2_2p
    # колено 2
    print("finding coordinates of knee2...")
    x2_1 = [find_knee(x0[i], y0[i], x2_2[i], y2_2[i])[0] for i in range(len(t))]
    y2_1 = [find_knee(x0[i], y0[i], x2_2[i], y2_2[i])[1] for i in range(len(t))]
    # голова
    x3 = x0 - L3 * sin(psi)
    y3 = y0 + L3 * cos(psi)

    # найдем углы
    print("counting angles...")
    alpha1, beta1, alpha2, beta2 = find_angles(x0, y0, x1_1, y1_1, x1_2, y1_2, x2_1, y2_1, x2_2, y2_2)
    # найдем энергию 
    # предстоит решить систему уравнений для поиска qi, ui
    # они пока задаются из прошлого решения :)
    print("counting energy...")
    q1 = np.radians(96 + 35 * sin(omega * t) + 15 * cos(omega * t) - 2 * sin(2 * omega * t) + 2 * cos(2 * omega * t))
    q2 = np.radians(96 - 35 * sin(omega * t) - 15 * cos(omega * t) - 2 * sin(2 * omega * t) + 2 * cos(2 * omega * t))
    u1 = np.radians(175 - 57 * sin(omega * t) - 50 * cos(omega * t) + 5 * sin(2 * omega * t) - 10 * cos(2 * omega * t))
    u2 = np.radians(
        -127 - 85 * sin(omega * t) + 70 * cos(omega * t) + 50 * sin(2 * omega * t) - 31 * cos(2 * omega * t))
    energy, Q = count_energy(alpha1, beta1, alpha2, beta2, psi, q1, q2, u1, u2)
    print("counting reactions...")
    R1_ver = np.zeros(len(t))
    R1_hor = np.zeros(len(t))
    R2_ver = np.zeros(len(t))
    R2_hor = np.zeros(len(t))
    for i in range(len(t)):
        # когда пятка1 стоит на земле
        if y1_2[i] < 0.01:
            R1_ver[i] = G * (2 * M1 * cos(alpha1[i]) + 2 * M2 * cos(beta1[i]) + M3 * cos(psi[i]))
            R1_hor[i] = G * (2 * M1 * sin(alpha1[i]) + 2 * M2 * sin(beta1[i]) + M3 * sin(psi[i]))
            # print("R1 =", R1_ver[i], R1_hor[i])
        # когда пятка2 стоит на земле
        if y2_2[i] < 0.01:
            R2_ver[i] = G * (2 * M1 * cos(alpha2[i]) + 2 * M2 * cos(beta2[i]) + M3 * cos(psi[i]))
            R2_hor[i] = G * (2 * M1 * sin(alpha2[i]) + 2 * M2 * sin(beta2[i]) + M3 * sin(psi[i]))
            # print("R2 =", R2_ver[i], R2_hor[i])
    # запись в файл
    f = open('track_energy_react.txt', 'w')
    try:
        # работа с файлом
        # f.write ("t x0 y0 x1_1 y1_1 x1_2 y1_2 x2_1 y2_1 x2_2 y2_2 x3 y3 
        # alpha1 beta1 alpha2 beta2 energy Q
        # R1_ver, R1_hor, R2_ver, R2_hor
        for i in range(len(t)):
            print("%.2f " % round(t[i], 2),
                  "%.2f " % round(x0[i], 2), "%.2f " % round(y0[i], 2),
                  "%.2f " % round(x1_1[i], 2), "%.2f " % round(y1_1[i], 2),
                  "%.2f " % round(x1_2[i], 2), "%.2f " % round(y1_2[i], 2),
                  "%.2f " % round(x2_1[i], 2), "%.2f " % round(y2_1[i], 2),
                  "%.2f " % round(x2_2[i], 2), "%.2f " % round(y2_2[i], 2),
                  "%.2f " % round(x3[i], 2), "%.2f " % round(y3[i], 2),
                  "%.2f " % round(alpha1[i], 2), "%.2f " % round(beta1[i], 2),
                  "%.2f " % round(alpha2[i], 2), "%.2f " % round(beta2[i], 2),
                  "%.2f " % round(psi[i], 2),
                  "%.2f " % round(energy[i], 2), "%.2f " % round(Q[i], 2),
                  "%.2f " % round(R1_ver[i], 2), "%.2f " % round(R1_hor[i], 2),
                  "%.2f " % round(R2_ver[i], 2), "%.2f " % round(R2_hor[i], 2),
                  file=f)
    finally:
        f.close()
    print("end")
