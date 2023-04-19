from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from IPython import display

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
Q = []

type_ = 1
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
    # all file
    for i in range(len(text)):
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
        Q.append(Qx[i] + Qy[i] + Qa1[i] + Qb1[i] + Qa2[i] + Qb2[i] + Qpsi[i])
finally:
    f.close()

fig = plt.figure()
ax = fig.add_subplot(212, autoscale_on=True, xlim=(-0.5, 7.5), ylim=(-0.5, 2.5))
ax.set_aspect('equal')
ax.grid()
ax_t = fig.add_subplot(211, autoscale_on=True)
ax_t.set_xticks([])
ax_t.set_yticks([])

line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
line3, = ax.plot([], [], 'o-', lw=2)
time_template = r'$Время$ = %.1fс'
time_text = ax_t.text(0.02, 0.9, '', transform=ax_t.transAxes)
energy_template = r'$Е_{потенц}$ = %.1fДж'
legend_1 = ax_t.text(0.02, 0.79, '', transform=ax_t.transAxes)
Q_template = r'$Q_{обобщ}$ = %.1fДж'
legend_2 = ax_t.text(0.02, 0.68, '', transform=ax_t.transAxes)
R1_ver_template = r'$R_{1y}$ = %.1fН'
legend_3 = ax_t.text(0.35, 0.9, '', transform=ax_t.transAxes)
R1_hor_template = r'$R_{1x}$ = %.1fН'
legend_4 = ax_t.text(0.35, 0.79, '', transform=ax_t.transAxes)
R2_ver_template = r'$R_{2y}$ = %.1fН'
legend_5 = ax_t.text(0.35, 0.68, '', transform=ax_t.transAxes)
R2_hor_template = r'$R_{2x}$ = %.1fН'
legend_6 = ax_t.text(0.35, 0.57, '', transform=ax_t.transAxes)
est1_template = r'$Est_{1}$ = %.1fН*м/с'
legend_7 = ax_t.text(0.68, 0.68, '', transform=ax_t.transAxes)
est3_template = r'$Est_{2}$ = %.1fН*рад/с'
legend_8 = ax_t.text(0.68, 0.57, '', transform=ax_t.transAxes)
omega1_template = r'$\Omega_{1}$ = %.1fрад'
legend_9 = ax_t.text(0.68, 0.9, '', transform=ax_t.transAxes)
mom12_template = r'$M_{12}$ = %.1fН*м'
legend_10 = ax_t.text(0.68, 0.79, '', transform=ax_t.transAxes)
Qy_template = r'$Q_{y}$ = %.1fН*м'
legend_11 = ax_t.text(0.02, 0.57, '', transform=ax_t.transAxes)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    time_text.set_text('')
    legend_1.set_text('')
    legend_2.set_text('')
    legend_3.set_text('')
    legend_4.set_text('')
    legend_5.set_text('')
    legend_6.set_text('')
    legend_7.set_text('')
    legend_8.set_text('')
    legend_9.set_text('')
    legend_10.set_text('')
    legend_11.set_text('')
    return line1, line2, line3, time_text, legend_1, legend_2, legend_3, legend_4, legend_5, legend_6, legend_7, legend_8, legend_9, legend_10, legend_11


def animate(i):
    thisx1 = [x0[i], x1_1[i], x1_2[i]]
    thisy1 = [y0[i], y1_1[i], y1_2[i]]
    thisx2 = [x0[i], x2_1[i], x2_2[i]]
    thisy2 = [y0[i], y2_1[i], y2_2[i]]
    thisx3 = [x0[i], x3[i]]
    thisy3 = [y0[i], y3[i]]

    line1.set_data(thisx1, thisy1)
    line2.set_data(thisx2, thisy2)
    line3.set_data(thisx3, thisy3)
    time_text.set_text(time_template % (i * dt))
    legend_1.set_text(energy_template % (energy_p[i]))
    legend_2.set_text(Q_template % (Q[i]))
    legend_3.set_text(R1_ver_template % (R1_ver[i]))
    legend_4.set_text(R1_hor_template % (R1_hor[i]))
    legend_5.set_text(R2_ver_template % (R2_ver[i]))
    legend_6.set_text(R2_hor_template % (R2_hor[i]))
    legend_7.set_text(est1_template % (est1[i]))
    legend_8.set_text(est3_template % (est3[i]))
    legend_9.set_text(omega1_template % (Omega1[i]))
    legend_10.set_text(mom12_template % (Mom12[i]))
    legend_11.set_text(Qy_template % (Qy[i]))
    return line1, line2, line3, time_text, legend_1, legend_2, legend_3, legend_4, legend_5, legend_6, legend_7, legend_8, legend_9, legend_10, legend_11


ani = animation.FuncAnimation(fig, animate, range(1, len(t)),
                              interval=dt * 1000, blit=True, init_func=init)
plt.show()

"""# uncomment this to save!
# saving to m4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=40, extra_args=['-vcodec', 'libx264'])
ani.save(gif, writer='Pillow')"""

plt.close()
