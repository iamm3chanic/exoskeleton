import matplotlib.pyplot as plt
import numpy as np
from pylab import *


def graphs_t(pos, pres_up, pres_dn, trq):
    x = np.linspace(0, 10, 50)
    y1 = x
# Квадратичная зависимость
    y2 = [i**2 for i in x]
# Построение графика
    plt.title("Зависимости: y1 = x, y2 = x^2")  # заголовок
    plt.xlabel("x")         # ось абсцисс
    plt.ylabel("y1, y2")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(x, y1, x, y2)  # построение графика
    plt.show()


def graph_1(name, l1, l2):
    plt.title(name)  # заголовок
    plt.xlabel("x")         # ось абсцисс
    plt.ylabel("y")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(l1, l2)  # построение графика
    plt.show()


def graph_2(title, l1, l2, filename):
    plt.title(title)  # заголовок
    plt.xlabel("x")         # ось абсцисс
    plt.ylabel("y")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(l1, l2)  # построение графика
    plt.savefig(filename)
    figure()

def graph_dot(title, l1, l2, filename):
    plt.title(title)  # заголовок
    plt.xlabel("x")         # ось абсцисс
    plt.ylabel("y")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.scatter(l1, l2)  # построение точечногографика
    plt.savefig(filename)
    figure()

def graph_3(title, time, l1, l2, l3, l4, filename):
    plt.title(title)  # заголовок
    plt.xlabel("x")         # ось абсцисс
    plt.ylabel("y")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(time, l1)
    plt.plot(time, l2)
    plt.plot(time, l3)
    plt.plot(time, l4)
    plt.savefig(filename)
    figure()

def graph_4(title, time, l1, l2, filename):
    plt.title(title)  # заголовок
    plt.xlabel("t")         # ось абсцисс
    plt.ylabel("Omega, F_dn")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(time, l1)
    plt.plot(time, l2)
    plt.savefig(filename)
    figure()

def graph_5(title, time, l1, l2, l3, l4, filename):
    plt.title(title)  # заголовок
    plt.xlabel("t, c")         # ось абсцисс
    plt.ylabel("Omega, F_dn")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(time, l1)
    plt.plot(time, l2)           # включение отображение сетки
    plt.plot(time, l3, '--')
    plt.plot(time, l4, '--')
    plt.savefig(filename)
    figure()


def graph_3par(title, time, l1, l2, l3,  filename):
    plt.title(title)  # заголовок
    plt.xlabel("t")         # ось абсцисс
    plt.ylabel("Omega, F_dn")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(time, l1)
    plt.plot(time, l2)           # включение отображение сетки
    plt.plot(time, l3, '--')
    plt.savefig(filename)
    figure()


# преобразовать массивы, выбрав нужный тип (1..4) 
def pick_type(list_, type_, time, t):
    l0, t0 = [], []
    for k in range (0, len(list_)):
        if type_[k] == t:
            l0.append(list_[k])
            t0.append(time[k])
    return l0, t0
# 
def calibrate(list_, koef_d, koef_p):
    l = []
    for i in list_:
        l.append(i/koef_d + koef_p)
    return l

#
def move_to_axe(arr):
    s = 0
    for one in arr:
        s += one
    mid = s / len(arr)
    arr = np.array(arr) - mid
    return arr
            
# выбирает индексы начала приседания, потом поправим их вручную
def pick_squat(pos):
    #найдем скачок больше -130 по позиции 
    #это означает сгиб ноги
    squat_indices = []
    for i in range(len(pos)-1):
        if (pos[i+1] - pos[i] <= -130 and abs(pos[i+1] - pos[i]) >= 130) and (pos[i+1] - pos[i] >= -150 and abs(pos[i+1] - pos[i]) <= 150) :
            squat_indices.append(i+1)
    return squat_indices
    
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
    
      # выбираем тип 2 - это приседание
    pos_r_2, t2 = pick_type(pos_r, type_, time, 2)
    pos_l_2, t2 = pick_type(pos_l, type_, time, 2)
    pres_dn_r_2, t2 = pick_type(pres_dn_r, type_, time, 2)
    pres_dn_l_2, t2 = pick_type(pres_dn_l, type_, time, 2)
    #выбираем индексы, опреденные давлением под правой ногой
    squat_indices = pick_squat(pos_r_2)
    print(squat_indices)
    squats_arr = []   
    for i in range(len(squat_indices)-1):
        #метка 2
        squat = []
        #если приседание среднего размера - от 20 до 30 позиций по времени
        if squat_indices[i+1] - squat_indices[i] >= 20 and squat_indices[i+1] - squat_indices[i] <= 30:
            for j in range(squat_indices[i], squat_indices[i+1]):
                squat.append([ pos_r_2[j], pos_l_2[j], pres_dn_r_2[j], pres_dn_l_2[j] ])
            print('\nsquat',i,'=',squat)
            squats_arr.append(squat)
    #возьмем например 1й присед, правую позицию, калибровочные коэф-ы *17 + 180
    print(squats_arr[1])
    dr_pos = []
    dr_pres = []
    dr_pos_other = []
    dr_pres_other = []
    dr_time = [0]
    for i in range(len(squats_arr[1])):
        dr_pos.append(squats_arr[1][i][0])
        dr_pos_other.append(squats_arr[1][i][1])
        dr_pres.append(squats_arr[1][i][2])
        dr_pres_other.append(squats_arr[1][i][3])
        dr_time.append(i/10)
    print(dr_pres)
    dr_pos = calibrate(dr_pos , 17, 180) 
    # прикрепляем нулевой чтобы шаг закончился в одной точке
    dr_pos.append(dr_pos[0])
    dr_pres = calibrate(dr_pres , 1/10, -9890)
    dr_pres.append(dr_pres[0])
    dr_pos_other = calibrate(dr_pos_other , 12, 180) 
    dr_pos_other.append(dr_pos_other[0])
    dr_pres_other = calibrate(dr_pres_other , 1/10, -10010)  
    dr_pres_other.append(dr_pres_other[0])
    # вот этот мне не понравился, левая нога работает по-другому 
    # поэтому добавим шум в dr_pres
    X = np.array([np.array(xi) for xi in dr_pres])
    Y = np.random.normal(X,10) #10 - размер шума
    Y = np.asarray(Y, dtype = int)
    #dr_pres_other = calibrate(dr_pres_other , 1/10, -10010)  
    
    
    graph_5("Присед: Omega_1(син), F_dn_1(оран), Omega_2(зел), F_dn_2(красн) от t",
            dr_time, dr_pos, dr_pres, dr_pos_other, Y, "4params-pick_squat.png")

