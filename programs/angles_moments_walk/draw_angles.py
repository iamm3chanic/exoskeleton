from numpy import sin, cos
from create_track import norm_list
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

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
est1, est2, est3 = [], [], []
Mom12, Mom22 = [], []
Omega1, Omega2 = [], []
f = open('track_energy_react.txt', 'r')

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
        Omega1.append(np.pi - alpha1[i] + beta1[i])
        Omega2.append((np.pi - alpha2[i] + beta2[i]) * 180 / np.pi)
        Mom12.append(Omega1[i] * R1y[i])
        Mom22.append(Omega1[i] * Qy[i])
finally:
    f.close()


# построение графика всех параметров от времени на одном графике
def graph_draw(title, t, data, filename, ylabel='angles, Рад',
               legend=None):
    if legend is None:
        legend = ['alpha1', 'alpha2', 'beta1', 'beta2', 'psi']
    plt.title(title)
    plt.xlabel("t, с")
    plt.ylabel(ylabel)
    plt.grid()
    for d in data:
        plt.plot(t, d)
    plt.legend(legend)
    plt.savefig(filename)
    plt.figure()
    plt.cla()


if __name__ == "__main__":
    graph_draw("Обобщенные координаты", t, [alpha1, alpha2, beta1, beta2, psi], "angles.png", ylabel='angles, Рад',
               legend=[r'$\alpha_1$', r'$\alpha_2$', r'$\beta_1$', r'$\beta_2$', r'$\psi$'])
    graph_draw("Обобщенные силы", t, [Qa1, Qa2, Qb1, Qb2, Qpsi], "forces.png", ylabel='forces, Н*м',
               legend=[r'$Q_{\alpha1}$', r'$Q_{\alpha2}$', r'$Q_{\beta1}$', r'$Q_{\beta2}$', r'$Q_{\psi}$'])
    graph_draw("Моменты", t, [u1, u2, q1, q2], "moments.png", ylabel='moments, Н*м',
               legend=[r'$М_{12}$', r'$М_{22}$', r'$М_{13}$', r'$М_{23}$'])
    graph_draw("Моменты в коленных суставах", t, [u1, u2], "moments_knee.png", ylabel='moments, Н*м',
               legend=[r'$М_{12}$', r'$М_{22}$'])
    graph_draw("Моменты в тазобедренном суставе", t, [q1, q2], "moments_corpus.png", ylabel='moments, Н*м',
               legend=[r'$М_{13}$', r'$М_{23}$'])
    graph_draw("Реакции", t, [R1_hor, R1_ver, R1x, R1y], "reactions.png", ylabel='reactions, Н',
               legend=[r'$R_{1x}$ статическая', r'$R_{1y}$ статическая', r'$R_{1x}$ динамическая',
                       r'$R_{1y}$ динамическая'])
    graph_draw("Момент в коленном суставе, реакция и разность углов", t, [u1, Mom12], "moment_angle.png",
               ylabel='moment+angle',
               legend=[r'$u_1$', r'$\Omega_1 R_{1y}$'])
    graph_draw("Момент в коленном суставе, Qy и разность углов", t, [u1, Mom22], "Qy_angle.png", ylabel='moment+angle',
               legend=[r'$u_1$', r'$\Omega_1 Q_y$'])
    graph_draw("Энергетические оценки для одного шарнира", t, [est1, est2], "estimations1.png",
               ylabel='estimations, Н*м/с',
               legend=[r"$M_{real} \Omega$', 424.3", r"$M_{real}^2 \Omega$', 3.53e5"])
    graph_draw("Энергетические оценки, реакция и разность углов", t, [est1, est3], "estimations2.png",
               ylabel='estimations', legend=[r"$M_{real} \Omega$', 424.3", r"$\Omega$ $R_{1y}$ $\Omega$', 909.0"])
    graph_draw("Шаг (теор): угол и сила реакции от времени", t, [[i * 180 / 3.14 for i in Omega1], R1y],
               "theor_step.png",
               ylabel=r"$\Omega, °$, $R_{y}, H$", legend=[r"$\Omega$", r"$R_{y}$"])
