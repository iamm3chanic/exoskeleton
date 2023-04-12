from dtw import *
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.spatial.distance import euclidean
from create_track import norm_list

G = 9.8  # ускорение свободного падения, м/с^2
L1 = 0.6  # длина бедра, м
L2 = 0.53  # длина голени, м
L3 = 0.75  # длина корпуса, м
M1 = 9.9  # масса бедра, кг
M2 = 7.2  # масса голени, кг
M3 = 25.  # масса корпуса, кг
h = 1.1  # высота точки подвеса
s = 0.2  # опорный сдвиг
L_step = 0.5  # длина шага
Ampl = 0.2  # амплитуда синусоиды движения переносной ноги
T = 1.1  # период двойного шага
omega = 2 * 3.14 / T  # угловая скорость
dt = 0.05  # изменение времени
# t = np.arange(0, 7, dt)

t = []
x0, y0, x1_1, y1_1, x1_2, y1_2, x2_1, y2_1, x2_2, y2_2, x3, y3 = [], [], [], [], [], [], [], [], [], [], [], []
alpha1, beta1, alpha2, beta2, psi, energy_p = [], [], [], [], [], []
Qx, Qy, Qpsi, Qa1, Qa2, Qb1, Qb2 = [], [], [], [], [], [], []
R1_ver, R1_hor, R2_ver, R2_hor = [], [], [], []
u1, u2, q1, q2 = [], [], [], []
R1x, R1y, R2x, R2y = [], [], [], []
est1, est2, est3, est4 = [], [], [], []
Mom12, Mom22 = [], []
Omega1, Omega2 = [], []

type_ = 3
filename = 'track_energy_react.txt'
if type_ == 1:
    filename = 'track_walk.txt'
    est_file = 'est_walk.txt'
    gif = 'walk.gif'
elif type_ == 2:
    filename = 'track_fast_walk.txt'
    est_file = 'est_fast_walk.txt'
    gif = 'fast_walk.gif'
elif type_ == 3:
    filename = 'track_run.txt'
    est_file = 'est_run.txt'
    gif = 'run.gif'

f = open(filename, 'r')
try:
    # работа с файлом
    text = f.readlines()
    # первые 1.1 секунд
    for i in range(23):
        string = text[i].split()
        t.append(float(string[0]))
        x0.append(float(string[1]))
        y0.append(float(string[2]))
        x1_1.append(float(string[3]))
        y1_1.append(float(string[4]))
        x1_2.append(float(string[5]))
        y1_2.append(float(string[6]))
        x2_1.append(float(string[7]))
        y2_1.append(float(string[8]))
        x2_2.append(float(string[9]))
        y2_2.append(float(string[10]))
        x3.append(float(string[11]))
        y3.append(float(string[12]))
        alpha1.append(float(string[13]))
        beta1.append(float(string[14]))
        alpha2.append(float(string[15]))
        beta2.append(float(string[16]))
        psi.append(float(string[17]))
        energy_p.append(float(string[18]))
        Qx.append(float(string[19]))
        Qy.append(float(string[20]))
        Qpsi.append(float(string[21]))
        Qa1.append(float(string[22]))
        Qa2.append(float(string[23]))
        Qb1.append(float(string[24]))
        Qb2.append(float(string[25]))
        R1_ver.append(float(string[26]))
        R1_hor.append(float(string[27]))
        R2_ver.append(float(string[28]))
        R2_hor.append(float(string[29]))
        u1.append(float(string[30]))
        u2.append(float(string[31]))
        q1.append(float(string[32]))
        q2.append(float(string[33]))
        R1x.append(float(string[34]))
        R1y.append(float(string[35]))
        R2x.append(float(string[36]))
        R2y.append(float(string[37]))
        est1.append(float(string[38]))
        est2.append(float(string[39]))
        est3.append(float(string[40]))
        est4.append(float(string[41]))
        Omega1.append(np.pi - alpha1[i] + beta1[i])
        Omega2.append((np.pi - alpha2[i] + beta2[i]) * 180 / np.pi)
        Mom12.append(Omega1[i] * R1y[i])
        Mom22.append(Omega1[i] * Qy[i])
finally:
    f.close()

f_est = open(est_file, 'r')
int1, int2, int3, int4 = 0, 0, 0, 0
try:
    text = f_est.readlines()
    int1 = float(text[0])
    int2 = float(text[1])
    int3 = float(text[2])
    int4 = float(text[3])
finally:
    f_est.close()


def dtw_pic(array1, array2, filename='dtw_u1_m12.png', ylabel='moment+angle',
            legend=[r'$M_{12}$', r'$\Omega_1 R_{1y}$'], title="Момент в коленном суставе, реакция и разность углов"):
    fig, ax = plt.subplots(figsize=(7, 5))
    # ???
    # array1 = norm_list(array1)
    # array2 = norm_list(array2)
    plt.xlabel("t, с")
    plt.ylabel(ylabel)
    plt.grid()
    array1 = np.reshape(np.array(array1), (len(array1), 1))
    array2 = np.reshape(np.array(array2), (len(array2), 1))
    dtw_distance, warp_path = fastdtw(array1, array2, dist=euclidean)
    for [map_x, map_y] in warp_path:
        ax.plot([map_x, map_y], [array1[map_x], array2[map_y]], '--k', linewidth=2)
    ax.plot(array1, color="red", label=legend[0], marker="o", linewidth=3, alpha=0.5)
    ax.plot(array2, color="blue", label=legend[1], marker="o", linewidth=3, alpha=0.5)
    ax.set_title(title)

    ax.set_xticks([0, 10, 20])
    ax.set_xticklabels(['0', '0.5', '1'])
    ax.legend()
    plt.savefig(filename)


dtw_pic(u1, Mom12)
dtw_pic(u1, Mom22, filename='dtw_u1_qy.png',
        legend=[r'$M_{12}$', r'$\Omega_1 Q_{y}$'], title="Момент в коленном суставе, $Q_y$ и разность углов")
dtw_pic(norm_list(est1), norm_list(est4), title="Энергетические оценки для одного шарнира", filename="dtw_estimations3.png",
        ylabel='estimations, Н*м/с',
        legend=[r"$M_{real} \Omega$', " + str(int1), r"$M_{real}$, " + str(int4)])
dtw_pic(norm_list(est1), norm_list(est3), title="Энергетические оценки, реакция и разность углов", filename="dtw_estimations2.png",
        ylabel='estimations',
        legend=[r"$M_{real} \Omega$', " + str(int1), r"$\Omega$ $R_{1y}$ $\Omega$', " + str(int3)])
