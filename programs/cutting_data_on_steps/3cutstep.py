import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.interpolate import krogh_interpolate
from pylab import *


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


def graph_4(title, time, l1, l2, filename):  # f1(t), f2(t)
    plt.title(title)  # заголовок
    plt.xlabel("t, c")  # ось абсцисс
    plt.ylabel("Omega, F_dn")  # ось ординат
    plt.grid()  # включение отображение сетки
    # !!! test
    # time = np.linspace(0, 2, 100)
    plt.plot(time, l1)
    plt.plot(time, l2)
    plt.savefig(filename)
    figure()


def graph_5(title, time, l1, l2, l3, l4, filename):
    plt.title(title)  # заголовок
    plt.xlabel("t, c")  # ось абсцисс
    plt.ylabel("Omega(°), F_dn(Н/10)")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(time, l1)
    plt.plot(time, l2)  # включение отображение сетки
    plt.plot(time, l3, '--')
    plt.plot(time, l4, '--')
    plt.savefig(filename)
    figure()


def graph_3par(title, time, l1, l2, l3, filename):
    plt.title(title)  # заголовок
    plt.xlabel("t, c")  # ось абсцисс
    plt.ylabel("Omega, F_dn")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(time, l1)
    plt.plot(time, l2)  # включение отображение сетки
    plt.plot(time, l3, '--')
    plt.savefig(filename)
    figure()


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


if __name__ == "__main__":
    time, pos_r, pos_l, pres_up_r, pres_up_l, pres_dn_r, pres_dn_l, trq_r, trq_l, type_ = [
    ], [], [], [], [], [], [], [], [], []
    f = open('1_2.txt', 'r')
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
            type_.append(int(string[9]))
            # str(f.readline(i+1)) #.split()
            # i+=1
    finally:
        f.close()

    # выбираем тип 1 - это шаги
    pos_r_1, t1 = pick_type(pos_r, type_, time, 1)
    pos_l_1, t1 = pick_type(pos_l, type_, time, 1)
    pres_dn_r_1, t1 = pick_type(pres_dn_r, type_, time, 1)
    pres_dn_l_1, t1 = pick_type(pres_dn_l, type_, time, 1)
    # выбираем индексы, опреденные давлением под правой ногой
    step_indices = pick_step(pres_dn_r_1)
    print(step_indices)
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
    print(steps_arr[k])
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
    print(dr_pres)
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

    ##### reshape
    # x = np.arange(0, len(dr_time))
    x = np.linspace(0, 2, len(dr_time))
    y = np.array(dr_pos)
    z = np.array(dr_pres)

    fy = interpolate.interp1d(x, y)
    fz = interpolate.interp1d(x, z)
    # step = reshape(step,20)
    x = np.linspace(0, 2, 39)  # для теста можно разные кол-ва точек
    ynew = fy(x)  # use interpolation function returned by `interp1d`
    # znew = fz(x)
    znew = krogh_interpolate(x, fz(x), x)
    znew = [i * 10 for i in znew]

    # ПРОДУМАТЬ ИНТЕРполяцию
    graph_draw("Шаг (эксп): угол и сила реакции от времени",
               x, [ynew, znew], "pick_step.png",
               ylabel=r"$\Omega, °$, $R_{y}, H$", legend=[r"$\Omega$", r"$R_{y}$"])

    graph_draw("Шаг (эксп): углы и силы реакции от времени",
               dr_time, [dr_pos_other, dr_pres_other, dr_pos, dr_pres], "4params-pick_step.png",
               ylabel=r"$\Omega, °$, $R_{y}, H$", legend=[r"$\Omega_1$", r"$R_{1y}$", r"$\Omega_2$", r"$R_{2y}$"])

    graph_draw("Шаг (эксп): угол и сила реакции от времени",
               dr_time, [dr_pos_other, [i * 5 for i in dr_pres_other]], "exp_step.png",
               ylabel=r"$\Omega, °$, $R_{y}, H$", legend=[r"$\Omega$", r"$R_{y}$"])
