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

time, pos_r, pos_l, pres_up_r, pres_up_l, pres_dn_r, pres_dn_l, trq_r, trq_l, type_ = [
], [], [], [], [], [], [], [], [], []
f_ = open('data_exp.txt', 'r')
try:
    # работа с файлом
    text = f_.readlines()
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
        type_.append(int(string[9]))
        # str(f.readline(i+1)) #.split()
        # i+=1
finally:
    f_.close()


def dtw_pic(array1, array2, filename='dtw_u1_m12.png', ylabel='moment+angle',
            legend=[r'$M_{12}$', r'$\Omega_1 R_{1y}$'], title="Момент в коленном суставе, реакция и разность углов", ylim=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    # ???
    # array1 = norm_list(array1)
    # array2 = norm_list(array2)
    plt.xlabel("t, с")
    plt.ylabel(ylabel)
    plt.grid()
    if ylim is not None:
        plt.ylim(ylim)
    #t1 = np.linspace(0, 22, np.shape(array1)[0])
    #t2 = np.linspace(0, 22, np.shape(array2)[0])
    array1 = np.reshape(np.array(array1), (len(array1), 1))
    #array2 = [i*1.1/0.95 for i in array2]
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


# преобразовать массивы, выбрав нужный тип (1..4)
def pick_type(list_, type_, time, t):
    l0, t0 = [], []
    for k in range(0, len(list_)):
        if type_[k] == t:
            l0.append(list_[k])
            t0.append(time[k])
    return l0, t0


#
def calibrate(list_, koef_d, koef_p):
    l = []
    for i in list_:
        l.append(i / koef_d + koef_p)
    return l


#
def move_to_axe(arr):
    s = 0
    for one in arr:
        s += one
    mid = s / len(arr)
    arr = np.array(arr) - mid
    return arr


# выбирает индексы начала шага, потом поправим их вручную
def pick_step(pres):
    # найдем скачок больше -20 по давлению
    # это означает поднятие ноги
    step_indices = []
    for i in range(len(pres) - 1):
        if pres[i + 1] - pres[i] <= -20 and abs(pres[i + 1] - pres[i]) >= 20:
            step_indices.append(i + 1)
    return step_indices


def exp_preparations():
    # выбираем тип 1 - это шаги
    pos_r_1, t1 = pick_type(pos_r, type_, time, 1)
    pos_l_1, t1 = pick_type(pos_l, type_, time, 1)
    pres_dn_r_1, t1 = pick_type(pres_dn_r, type_, time, 1)
    pres_dn_l_1, t1 = pick_type(pres_dn_l, type_, time, 1)
    # выбираем индексы, опреденные давлением под правой ногой
    step_indices = pick_step(pres_dn_r_1)
    # print(step_indices)
    steps_arr = []
    for i in range(len(step_indices) - 1):
        step = []
        # если шаг среднего размера - от 19 до 25 позиций по времени
        if step_indices[i + 1] - step_indices[i] >= 19 and step_indices[i + 1] - step_indices[i] <= 25:
            for j in range(step_indices[i], step_indices[i + 1]):
                step.append([pos_r_1[j], pos_l_1[j], pres_dn_r_1[j], pres_dn_l_1[j]])
            # print('\nstep',i,'=',step)
            steps_arr.append(step)
    # возьмем например Kй шаг, правую позицию, калибровочные коэф-ы *17 + 180
    k = 4
    # print(steps_arr[k])
    dr_pos = []
    dr_pres = []
    dr_pos_other = []
    # dr_pres_other = [1011]
    dr_pres_other = []
    dr_time = [0]
    for i in range(len(steps_arr[k])):
        dr_pos.append(steps_arr[k][i][0])
        dr_pos_other.append(steps_arr[k][i][1])
        dr_pres.append(steps_arr[k][i][2])
        dr_pres_other.append(steps_arr[k][i][3])
        dr_time.append(i / 10)
    # print(dr_pres)
    dr_pos = calibrate(dr_pos, 17, 180)
    # прикрепляем нулевой чтобы шаг закончился в одной точке
    dr_pos.append(dr_pos[0])
    dr_pres = calibrate(dr_pres, 1, -989)
    dr_pres.append(dr_pres[0])
    dr_pos_other = calibrate(dr_pos_other, 17, 180)
    dr_pos_other.append(dr_pos_other[0])
    dr_pres_other = calibrate(dr_pres_other, 1, -1001)
    # первая мне не нравится
    # dr_pres_other.pop(1)
    dr_pres_other.append(dr_pres_other[0])
    return dr_time, dr_pos_other, dr_pres_other


dtw_pic(u1, Mom12)
dtw_pic(u1, Mom22, filename='dtw_u1_qy.png',
        legend=[r'$M_{12}$', r'$\Omega_1 Q_{y}$'], title="Момент в коленном суставе, $Q_y$ и разность углов")
dtw_pic(norm_list(est1), norm_list(est4), title="Энергетические оценки для одного шарнира",
        filename="dtw_estimations3.png",
        ylabel='estimations, Н*м/с',
        legend=[r"$M_{real} \Omega$', " + str(int1), r"$M_{real}$, " + str(int4)])
dtw_pic(norm_list(est1), norm_list(est3), title="Энергетические оценки, реакция и разность углов",
        filename="dtw_estimations2.png",
        ylabel='estimations',
        legend=[r"$M_{real} \Omega$', " + str(int1), r"$\Omega$ $R_{1y}$ $\Omega$', " + str(int3)])

a, b, c = exp_preparations()
b = np.radians(b)
#b = [i*1.1/0.95  for i in b]
c = [i * 5  for i in c]
dtw_pic(Omega1, b, title="Шаг (угол): сравнение эксперимента и теории", filename="dtw_angles.png",
        ylabel=r"$\Omega, рад$",
        legend=[r"$\Omega_Т$", r"$\Omega_Э$"], ylim=(1, 3.5))
dtw_pic(R1y, c, title="Шаг (сила реакции): сравнение эксперимента и теории", filename="dtw_forces.png",
        ylabel=r"$R_{1y}, H$",
        legend=[r"$R_Т$", r"$R_Э$"])
