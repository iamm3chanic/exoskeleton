import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from pylab import *
import csv


# график зависимости l2 от l1 с выводом в файл
def graph_2(title, l1, l2, filename):
    plt.title(title)  # заголовок
    plt.xlabel("x")  # ось абсцисс
    plt.ylabel("y")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(l1, l2)  # построение графика
    plt.savefig(filename)
    figure()


# чтение файла и запись данных в массив
def read_file(filename):
    time, pos_r, pos_l, pres_up_r, pres_up_l, pres_dn_r, pres_dn_l, trq_r, trq_l, type_ = [], [], [], [], [], [], [], [], [], []
    f = open(filename, 'r')
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
    finally:
        f.close()
    return time, pos_r, pos_l, pres_up_r, pres_up_l, pres_dn_r, pres_dn_l, trq_r, trq_l, type_


# преобразовать массивы, выбрав нужный тип (1..4)
def pick_type(list_, type_, time, t):
    l0, t0 = [], []
    for k in range(0, len(list_)):
        if type_[k] == t:
            l0.append(list_[k])
            t0.append(time[k])
    return l0, t0


#
def calibrate(list_, koef):
    l = []
    for i in list_:
        l.append(i / koef)
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


# выбирает индексы начала приседания, потом поправим их вручную
def pick_squat(pos):
    # найдем скачок больше -130 по позиции
    # это означает сгиб ноги
    squat_indices = []
    for i in range(len(pos) - 1):
        if (pos[i + 1] - pos[i] <= -130 and abs(pos[i + 1] - pos[i]) >= 130) and (
                pos[i + 1] - pos[i] >= -150 and abs(pos[i + 1] - pos[i]) <= 150):
            squat_indices.append(i + 1)
    return squat_indices


# выбирает индексы начала шага, потом поправим их вручную
def pick_stair(pres):
    # найдем скачок больше -20 по давлению
    # это означает поднятие ноги
    stair_indices = []
    for i in range(len(pres) - 1):
        if pres[i + 1] - pres[i] <= -20 and abs(pres[i + 1] - pres[i]) >= 20:
            stair_indices.append(i + 1)
    return stair_indices


##############################################

if __name__ == "__main__":
    time, pos_r, pos_l, pres_up_r, pres_up_l, pres_dn_r, pres_dn_l, trq_r, trq_l, type_ = read_file(
        'test30_09_nums_only.txt')

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
        # метка 1 (временно убрали)
        step = []
        # если шаг среднего размера - от 19 до 25 позиций по времени
        if step_indices[i + 1] - step_indices[i] >= 19 and step_indices[i + 1] - step_indices[i] <= 25:
            for j in range(step_indices[i], step_indices[i + 1]):
                step.append([pos_r_1[j], pos_l_1[j], pres_dn_r_1[j], pres_dn_l_1[j]])
            print('\nstep', i, '=', step)

            # reshape
            x = np.arange(0, len(step) * 4)
            y = np.array(step).reshape(len(step) * 4)
            print(np.asarray(x).shape, np.asarray(y).shape)
            f = interpolate.interp1d(x, y)
            # step = reshape(step,20)
            xnew = np.arange(0, 60)
            ynew = f(xnew)  # use interpolation function returned by `interp1d`

            ynew = np.asarray(ynew, dtype=int)
            # метка 1
            ynew = np.append(ynew, 1)

            steps_arr.append(ynew)

            # add noise
            for i in range(7):
                X = np.array([np.array(xi) for xi in ynew])
                Y = np.random.normal(X, 5)  # 5 - размер шума
                Y = np.asarray(Y, dtype=int)  # .reshape(60,1)
                Y = np.delete(Y, 60)
                Y = np.append(Y, 1)
                steps_arr.append(Y)

            # print(X, Y)
            # X += Y
            # Y = 60*np.random.randn(100)
            # noise = np.random.normal(0, 1, 100)
            # data = np.hstack((X.reshape(100,1),Y.reshape(100,1)))

    myFile = open('example1.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(steps_arr)

    print("Writing 1 complete")

    # выбираем тип 2 - это приседание
    pos_r_2, t2 = pick_type(pos_r, type_, time, 2)
    pos_l_2, t2 = pick_type(pos_l, type_, time, 2)
    pres_dn_r_2, t2 = pick_type(pres_dn_r, type_, time, 2)
    # pres_dn_l_2, t2 = pick_type(pres_dn_l, type_, time, 2)
    pres_dn_l_2 = np.array([np.array(xi) for xi in pres_dn_r_2])
    Y = np.random.normal(pres_dn_l_2, 5)  # 5 - размер шума
    pres_dn_l_2 = np.asarray(Y, dtype=int)
    # выбираем индексы, опреденные давлением под правой ногой
    squat_indices = pick_squat(pos_r_2)
    print(squat_indices)
    squats_arr = []
    for i in range(len(squat_indices) - 1):
        # метка 2 (временно убрали)
        squat = []
        # если приседание среднего размера - от 20 до 30 позиций по времени
        if squat_indices[i + 1] - squat_indices[i] >= 20 and squat_indices[i + 1] - squat_indices[i] <= 30:
            for j in range(squat_indices[i], squat_indices[i + 1]):
                squat.append([pos_r_2[j], pos_l_2[j], pres_dn_r_2[j], pres_dn_l_2[j]])
            print('\nsquat', i, '=', squat)

            ##### reshape
            x = np.arange(0, len(squat) * 4)
            y = np.array(squat).reshape(len(squat) * 4)
            print(np.asarray(x).shape, np.asarray(y).shape)
            f = interpolate.interp1d(x, y)
            # step = reshape(step,20)
            xnew = np.arange(0, 60)
            ynew = f(xnew)  # use interpolation function returned by `interp1d`

            ynew = np.append(ynew, 2)
            ynew = np.asarray(ynew, dtype=int)

            squats_arr.append(ynew)
            # add noise
            for i in range(10):
                X = np.array([np.array(xi) for xi in ynew])
                Y = np.random.normal(X, 5)  # 5 - размер шума
                Y = np.asarray(Y, dtype=int)  # .reshape(60,1)
                Y = np.delete(Y, 60)
                Y = np.append(Y, 2)
                squats_arr.append(Y)
    # print('\nsteps_arr =',steps_arr)

    myFile = open('example2.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(squats_arr)

    print("Writing 2 complete")

    # выбираем тип 4 - это шаги (в исходном файле 4!)
    pos_r_3, t3 = pick_type(pos_r, type_, time, 4)
    pos_l_3, t3 = pick_type(pos_l, type_, time, 4)
    pres_dn_r_3, t3 = pick_type(pres_dn_r, type_, time, 4)
    pres_dn_l_3, t3 = pick_type(pres_dn_l, type_, time, 4)
    # выбираем индексы, опреденные давлением под правой ногой
    stair_indices = pick_stair(pres_dn_r_3)
    print(stair_indices)
    stairs_arr = []
    for i in range(len(stair_indices) - 1):
        # метка 3 (временно убрали)
        stair = []
        # если шаг среднего размера - от 19 до 25 позиций по времени
        if stair_indices[i + 1] - stair_indices[i] >= 28 and stair_indices[i + 1] - stair_indices[i] <= 51:
            for j in range(stair_indices[i], stair_indices[i + 1]):
                stair.append([pos_r_3[j], pos_l_3[j], pres_dn_r_3[j], pres_dn_l_3[j]])
            print('\nstair', i, '=', stair)

            ##### reshape
            x = np.arange(0, len(stair) * 4)
            y = np.array(stair).reshape(len(stair) * 4)
            print(np.asarray(x).shape, np.asarray(y).shape)
            f = interpolate.interp1d(x, y)
            # step = reshape(step,20)
            xnew = np.arange(0, 60)
            ynew = f(xnew)  # use interpolation function returned by `interp1d`

            ynew = np.append(ynew, 3)
            ynew = np.asarray(ynew, dtype=int)

            stairs_arr.append(ynew)
            # add noise
            for i in range(10):
                X = np.array([np.array(xi) for xi in ynew])
                Y = np.random.normal(X, 5)  # 5 - размер шума
                Y = np.asarray(Y, dtype=int)  # .reshape(60,1)
                Y = np.delete(Y, 60)
                Y = np.append(Y, 3)
                stairs_arr.append(Y)

    myFile = open('example3.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(stairs_arr)

    print("Writing 3 complete")

    '''
    один шаг - все данные между двумя нулями давлений
    примерно 20 строк, если брать давление и позицию с каждой ноги это 20*4 = 80
    возьмем каждую строку за кортеж(список) из 4х позиций, остается только список длины 20
    этот список и есть один шаг
    похожие списки из кортежей мы будем анализировать на предмет того, шаг ли это
    '''
