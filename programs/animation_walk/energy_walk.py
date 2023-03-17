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
t = np.arange(0, 7, dt)

x0, y0, x1_1, y1_1, x1_2, y1_2, x2_1, y2_1, x2_2, y2_2, x3, y3 = [], [], [], [], [], [], [], [], [], [], [], []
alpha1, beta1, alpha2, beta2, psi, energy, Q = [], [], [], [], [], [], []
R1_ver, R1_hor, R2_ver, R2_hor = [], [], [], []
# f = open('track.txt', 'r')
f = open('track_energy_react.txt', 'r')

try:
    # работа с файлом
    text = f.readlines()
    for item in text:
        string = item.split()
        # t.append(int(string[0]))
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
        energy.append(float(string[18]))
        Q.append(float(string[19]))
        R1_ver.append(float(string[20]))
        R1_hor.append(float(string[21]))
        R2_ver.append(float(string[22]))
        R2_hor.append(float(string[23]))
finally:
    f.close()

# energy = np.zeros(len(t))
# Q = np.zeros(len(t))

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
time_template = 'Время = %.1fс'
time_text = ax_t.text(0.02, 0.9, '', transform=ax_t.transAxes)
energy_template = 'П_энергия = %.1fДж'
legend_1 = ax_t.text(0.02, 0.79, '', transform=ax_t.transAxes)
Q_template = 'Q_обобщ = %.1fДж'
legend_2 = ax_t.text(0.02, 0.68, '', transform=ax_t.transAxes)
R1_ver_template = 'Реакция_1_верт = %.1fН'
legend_3 = ax_t.text(0.35, 0.9, '', transform=ax_t.transAxes)
R1_hor_template = 'Реакция_1_гор = %.1fН'
legend_4 = ax_t.text(0.35, 0.79, '', transform=ax_t.transAxes)
R2_ver_template = 'Реакция_2_верт = %.1fН'
legend_5 = ax_t.text(0.35, 0.68, '', transform=ax_t.transAxes)
R2_hor_template = 'Реакция_2_гор = %.1fН'
legend_6 = ax_t.text(0.35, 0.57, '', transform=ax_t.transAxes)


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
    return line1, line2, line3, time_text, legend_1, legend_2, legend_3, legend_4, legend_5, legend_6


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
    legend_1.set_text(energy_template % (energy[i]))
    legend_2.set_text(Q_template % (Q[i]))
    legend_3.set_text(R1_ver_template % (R1_ver[i]))
    legend_4.set_text(R1_hor_template % (R1_hor[i]))
    legend_5.set_text(R2_ver_template % (R2_ver[i]))
    legend_6.set_text(R2_hor_template % (R2_hor[i]))
    return line1, line2, line3, time_text, legend_1, legend_2, legend_3, legend_4, legend_5, legend_6


ani = animation.FuncAnimation(fig, animate, range(1, len(t)),
                              interval=dt * 1000, blit=True, init_func=init)
plt.show()
"""
# uncomment this to save!
# saving to m4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=20, extra_args=['-vcodec', 'libx264'])
ani.save('react_energy_walk.gif', writer='Pillow')
"""
plt.close()
