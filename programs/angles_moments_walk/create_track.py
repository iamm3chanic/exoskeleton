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
M = M1 + M2 + M3  # общая масса
h = 1.11  # высота точки подвеса
s = 0.2  # опорный сдвиг
L_step = 0.5  # длина шага
Ka = L1 * (M1 + 2 * M2)  # момент бедра
Kb = M2 * L2  # момент голени
Kr = M3 * L3  # момент корпуса
Ja0 = M1 * L1 ** 2 / 3  # момент инерции бедра относительно оси z в т. О
Ja = Ja0 + 4 * M2 * L1 ** 2  # момент инерции бедра относительно оси z в т. колена
Jb = M2 * L2 ** 2 / 3  # момент инерции голени относительно оси z в т. колена
J = M3 * L3 ** 2 / 3  # момент инерции корпуса относительно оси z в т. О
Jab = 2 * M2 * L1 * L2  # относительный момент голень-бедро
Ampl = 0.2  # амплитуда синусоиды движения переносной ноги
T = 1.1  # период двойного шага
omega = 2 * 3.14 / T  # угловая скорость
dt = 0.05  # изменение времени
t = np.arange(0, 7, dt)


# первая производная
def deriv1(arr, delta=dt):
    res = []
    length = len(arr)
    # в первом элементе - правая производная
    res.append((arr[1] - arr[0]) / delta)
    # во внутренних элементах берем двустороннюю разность
    for i in range(1, length - 1):
        res.append((arr[i + 1] - arr[i - 1]) / (2 * delta))
    # в последнем - левая
    res.append((arr[length - 1] - arr[length - 2]) / delta)
    return np.array(res)


# первая производная
def deriv2(arr, delta=dt):
    res = deriv1(deriv1(arr, delta), delta)
    return res


# поиск колена по координатам пятки и таза
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


# поиск углов alpha1, beta1, alpha2, beta2 по координатам
def find_angles(x0, y0, x1_1, y1_1, x1_2, y1_2, x2_1, y2_1, x2_2, y2_2):
    alpha1, beta1, alpha2, beta2 = [], [], [], []
    for i in range(len(x0)):
        # бедро1 - вертикаль
        alpha1.append(-np.arctan((x1_1[i] - x0[i]) / (y1_1[i] - y0[i])))
        # голень1 - вертикаль
        beta1.append(-np.arctan((x1_2[i] - x1_1[i]) / (y1_2[i] - y1_1[i])))
        # бедро2 - вертикаль
        alpha2.append(-np.arctan((x2_1[i] - x0[i]) / (y2_1[i] - y0[i])))
        # голень2 - вертикаль
        beta2.append(-np.arctan((x2_2[i] - x2_1[i]) / (y2_2[i] - y2_1[i])))
    return alpha1, beta1, alpha2, beta2


# потенциальная и кинетическая энергия звеньев и всего аппарата по углам и моментам
def count_energy(alpha1, beta1, alpha2, beta2, psi):
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

    """ 
    # теперь моменты ищем оттуда, а не обратно
    # кинетическая энергия (обобщенные силы): сначала прибавим Q от корпуса
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
    Q = Q1 + Q2 + Q3 + Q4 + Q5"""

    return energy


def count_common_Q(x0, y0, alpha1, beta1, alpha2, beta2, psi):
    alpha1 = np.array(alpha1)
    beta1 = np.array(beta1)
    alpha2 = np.array(alpha2)
    beta2 = np.array(beta2)
    psi = np.array(psi)
    Qx = M * deriv2(x0) + Kr * (deriv2(psi) * cos(psi) - (deriv1(psi)) ** 2 * sin(psi)) + \
         Ka * (deriv2(alpha1) * cos(alpha1) - (deriv1(alpha1)) ** 2 * sin(alpha1)) + \
         Ka * (deriv2(alpha2) * cos(alpha2) - (deriv1(alpha2)) ** 2 * sin(alpha2)) + \
         Kb * (deriv2(beta1) * cos(beta1) - (deriv1(beta1)) ** 2 * sin(beta1)) + \
         Kb * (deriv2(beta2) * cos(beta2) - (deriv1(beta2)) ** 2 * sin(beta2))

    Qy = M * G + M * deriv2(y0) - Kr * (deriv2(psi) * sin(psi) + (deriv1(psi)) ** 2 * cos(psi)) + \
         Ka * (deriv2(alpha1) * sin(alpha1) + (deriv1(alpha1)) ** 2 * cos(alpha1)) + \
         Ka * (deriv2(alpha2) * sin(alpha2) + (deriv1(alpha2)) ** 2 * cos(alpha2)) + \
         Kb * (deriv2(beta1) * sin(beta1) + (deriv1(beta1)) ** 2 * cos(beta1)) + \
         Kb * (deriv2(beta2) * sin(beta2) + (deriv1(beta2)) ** 2 * cos(beta2))

    Qpsi = J * deriv2(psi) - Kr * (deriv2(y0) * sin(psi) - deriv2(x0) * cos(psi)) - \
           G * Kr * sin(psi)

    Qa1 = Ja * deriv2(alpha1) + Jab * deriv2(beta1) * cos(alpha1 - beta1) + \
          Ka * (deriv2(x0) * cos(alpha1) + deriv2(y0) * sin(alpha1)) + \
          Jab * (deriv1(beta1)) ** 2 * sin(alpha1 - beta1) + G * Ka * sin(alpha1)

    Qa2 = Ja * deriv2(alpha2) + Jab * deriv2(beta2) * cos(alpha2 - beta2) + \
          Ka * (deriv2(x0) * cos(alpha2) + deriv2(y0) * sin(alpha2)) + \
          Jab * (deriv1(beta2)) ** 2 * sin(alpha2 - beta2) + G * Ka * sin(alpha2)

    Qb1 = Jb * deriv2(beta1) + Jab * deriv2(alpha1) * cos(alpha1 - beta1) + \
          Kb * (deriv2(x0) * cos(beta1) + deriv2(y0) * sin(beta1)) - \
          Jab * (deriv1(alpha1)) ** 2 * sin(alpha1 - beta1) + G * Kb * sin(beta1)

    Qb2 = Jb * deriv2(beta2) + Jab * deriv2(alpha2) * cos(alpha2 - beta2) + \
          Kb * (deriv2(x0) * cos(beta2) + deriv2(y0) * sin(beta2)) - \
          Jab * (deriv1(alpha2)) ** 2 * sin(alpha2 - beta2) + G * Kb * sin(beta2)

    return Qx, Qy, Qpsi, Qa1, Qa2, Qb1, Qb2


# реакции опоры (статика)
def count_reactions(alpha1, beta1, alpha2, beta2, psi, Qx, Qy):
    R1_ver = np.zeros(len(t))
    R1_hor = np.zeros(len(t))
    R2_ver = np.zeros(len(t))
    R2_hor = np.zeros(len(t))
    R1y = np.zeros(len(t))
    R1x = np.zeros(len(t))
    R2y = np.zeros(len(t))
    R2x = np.zeros(len(t))
    for i in range(6):
        R1_ver[(i * 22):(i * 22) + 11] = G * (2 * M1 * cos(alpha1[(i * 22):(i * 22) + 11]) + 2 * M2 * cos(
            beta1[(i * 22):(i * 22) + 11]) + M3 * cos(psi[(i * 22):(i * 22) + 11]))
        R1_hor[(i * 22):(i * 22) + 11] = G * (2 * M1 * sin(alpha1[(i * 22):(i * 22) + 11]) + 2 * M2 * sin(
            beta1[(i * 22):(i * 22) + 11]) + M3 * sin(psi[(i * 22):(i * 22) + 11]))
        R1x[(i * 22):(i * 22) + 11] = Qx[(i * 22):(i * 22) + 11]
        R1y[(i * 22):(i * 22) + 11] = Qy[(i * 22):(i * 22) + 11]
    """for i in range(len(t)):
        # когда пятка1 стоит на земле
        if y1_2[i] < 0.01:
            R1_ver[i] = G * (2 * M1 * cos(alpha1[i]) + 2 * M2 * cos(beta1[i]) + M3 * cos(psi[i]))
            R1_hor[i] = G * (2 * M1 * sin(alpha1[i]) + 2 * M2 * sin(beta1[i]) + M3 * sin(psi[i]))
            R1x[i] = Qx[i]
            R1y[i] = Qy[i]
            # print("R1 =", R1_ver[i], R1_hor[i])
        # когда пятка2 стоит на земле
        if y2_2[i] < 0.01:
            R2_ver[i] = G * (2 * M1 * cos(alpha2[i]) + 2 * M2 * cos(beta2[i]) + M3 * cos(psi[i]))
            R2_hor[i] = G * (2 * M1 * sin(alpha2[i]) + 2 * M2 * sin(beta2[i]) + M3 * sin(psi[i]))
            R2x[i] = Qx[i]
            R2y[i] = Qy[i]
            # print("R2 =", R2_ver[i], R2_hor[i])"""
    return R1_ver, R1_hor, R2_ver, R2_hor, R1y, R1x, R2y, R2x


def find_moments(alpha1, beta1, Qx, Qy, Qpsi, Qa1, Qa2, Qb1, Qb2):
    # реакции и момент в стопе переносной ноги
    R2x, R2y, M21 = np.zeros(len(alpha1)), np.zeros(len(alpha1)), np.zeros(len(alpha1))
    # реакции и момент в стопе опорной ноги
    R1x, R1y = Qx, Qy
    # из ур-я Qa1 + Qb1
    # M13plusM23 = -Qpsi
    M13minusM11 = -(Qa1 + Qb1) + L1 * (R1x * cos(alpha1) + R1y * sin(alpha1)) + L2 * (
            R1x * cos(beta1) + R1y * sin(beta1))
    # M23minusM21 = -(Qa2 + Qb2)
    # пер корпус
    M23 = -(Qa2 + Qb2)
    # оп корпус
    M13 = -Qpsi - M23
    # оп стопа
    M11 = M13 - M13minusM11
    # оп колено
    M12 = -M11 + Qb1 - L2 * (R1x * cos(beta1) + R1y * sin(beta1))
    # пер колено
    M22 = -M21 + Qb2

    # сделать по полпериода
    m12, m22, m13, m23 = M12, M22, M13, M23
    for i in range(6):
        m12[(i * 22):(i * 22) + 11] = M12[:11]
        m12[(i * 22 + 11):(i * 22) + 22] = M22[:11]
        m22[(i * 22):(i * 22) + 11] = M22[:11]
        m22[(i * 22 + 11):(i * 22) + 22] = M12[:11]
        m13[(i * 22):(i * 22) + 11] = M13[:11]
        m13[(i * 22 + 11):(i * 22) + 22] = M23[:11]
        m23[(i * 22):(i * 22) + 11] = M23[:11]
        m23[(i * 22 + 11):(i * 22) + 22] = M13[:11]


    return M11, M21, m12, m22, m13, m23



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
    # пятка 2
    x2_2 = t + L_step / np.pi * (- np.sin(omega * t - np.pi))
    y2_2 = L_step * 0.1 * (1 - np.cos(omega * t - np.pi))
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

    # energy, Q = count_energy(alpha1, beta1, alpha2, beta2, psi, q1, q2, u1, u2)
    Qx, Qy, Qpsi, Qa1, Qa2, Qb1, Qb2 = count_common_Q(x0, y0, alpha1, beta1, alpha2, beta2, psi)
    energy_p = count_energy(alpha1, beta1, alpha2, beta2, psi)
    # найдем моменты
    # решая систему уравнений для поиска qi, ui
    print("counting moments...")
    # M11, M21, M12, M22, M13, M23 = w1, w2, u1, u2, q1, q2
    w1, w2, u1, u2, q1, q2 = find_moments(alpha1, beta1, Qx, Qy, Qpsi, Qa1, Qa2, Qb1, Qb2)
    """q1 = np.radians(96 + 35 * sin(omega * t) + 15 * cos(omega * t) - 2 * sin(2 * omega * t) + 2 * cos(2 * omega * t))
    q2 = np.radians(96 - 35 * sin(omega * t) - 15 * cos(omega * t) - 2 * sin(2 * omega * t) + 2 * cos(2 * omega * t))
    u1 = np.radians(175 - 57 * sin(omega * t) - 50 * cos(omega * t) + 5 * sin(2 * omega * t) - 10 * cos(2 * omega * t))
    u2 = np.radians(
        -127 - 85 * sin(omega * t) + 70 * cos(omega * t) + 50 * sin(2 * omega * t) - 31 * cos(2 * omega * t))"""
    print("counting reactions...")
    R1_ver, R1_hor, R2_ver, R2_hor, R1y, R1x, R2y, R2x = count_reactions(alpha1, beta1, alpha2, beta2, psi, Qx, Qy)

    # запись в файл
    f = open('track_energy_react.txt', 'w')
    try:
        # работа с файлом
        # f.write ("t x0 y0 x1_1 y1_1 x1_2 y1_2 x2_1 y2_1 x2_2 y2_2 x3 y3 
        # alpha1 beta1 alpha2 beta2 psi energy_p
        # Qx, Qy, Qpsi, Qa1, Qa2, Qb1, Qb2
        # R1_ver, R1_hor, R2_ver, R2_hor
        # u1, u2, q1, q2,
        # R1x, R1y, R2x, R2y")
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
                  "%.2f " % round(energy_p[i], 2),
                  "%.2f " % round(Qx[i], 2), "%.2f " % round(Qy[i], 2),
                  "%.2f " % round(Qpsi[i], 2),
                  "%.2f " % round(Qa1[i], 2), "%.2f " % round(Qa2[i], 2),
                  "%.2f " % round(Qb1[i], 2), "%.2f " % round(Qb2[i], 2),
                  "%.2f " % round(R1_ver[i], 2), "%.2f " % round(R1_hor[i], 2),
                  "%.2f " % round(R2_ver[i], 2), "%.2f " % round(R2_hor[i], 2),
                  "%.2f " % round(u1[i], 2), "%.2f " % round(u2[i], 2),
                  "%.2f " % round(q1[i], 2), "%.2f " % round(q2[i], 2),
                  "%.2f " % round(R1x[i], 2), "%.2f " % round(R1y[i], 2),
                  "%.2f " % round(R2x[i], 2), "%.2f " % round(R2y[i], 2),
                  file=f)
    finally:
        f.close()
    print("end")
