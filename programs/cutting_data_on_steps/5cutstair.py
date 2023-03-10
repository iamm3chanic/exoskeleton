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
            

# выбирает индексы начала шага, потом поправим их вручную
def pick_stair(pres):
    #найдем скачок больше -20 по давлению 
    #это означает поднятие ноги
    stair_indices = []
    for i in range(len(pres)-1):
        if pres[i+1] - pres[i] <= -20 and abs(pres[i+1] - pres[i]) >= 20:
            stair_indices.append(i+1)
    return stair_indices            
    
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
    
    # выбираем тип 4 - это ступеньки
    pos_r_3, t3 = pick_type(pos_r, type_, time, 4)
    pos_l_3, t3 = pick_type(pos_l, type_, time, 4)
    pres_dn_r_3, t3 = pick_type(pres_dn_r, type_, time, 4)
    pres_dn_l_3, t3 = pick_type(pres_dn_l, type_, time, 4)
    #выбираем индексы, опреденные давлением под правой ногой
    stair_indices = pick_stair(pres_dn_r_3)
    print(stair_indices)
    stairs_arr = []   
    for i in range(len(stair_indices)-1):
        #метка 3
        stair = []
        #если шаг среднего размера - от 19 до 25 позиций по времени
        if stair_indices[i+1] - stair_indices[i] >= 28 and stair_indices[i+1] - stair_indices[i] <= 51:
            for j in range(stair_indices[i], stair_indices[i+1]):
                stair.append([ pos_r_3[j], pos_l_3[j], pres_dn_r_3[j], pres_dn_l_3[j] ])
            print('\nstair',i,'=',stair)
            stairs_arr.append(stair)
            
    #возьмем например 1й шаг, правую позицию, калибровочные коэф-ы *17 + 180
    print(stairs_arr[1])
    dr_pos = []
    dr_pres = []
    dr_pos_other = []
    dr_pres_other = []
    dr_time = []
    for i in range(len(stairs_arr[1])):
        dr_pos.append(stairs_arr[1][i][0])
        dr_pos_other.append(stairs_arr[1][i][1])
        dr_pres.append(stairs_arr[1][i][2])
        dr_pres_other.append(stairs_arr[1][i][3])
        dr_time.append(i/10)
    #print(dr_pres)
    dr_pos = calibrate(dr_pos , 17, 180) 
    dr_pres = calibrate(dr_pres , 1, -989)
    dr_pos_other = calibrate(dr_pos_other , 17, 180) 
    dr_pres_other = calibrate(dr_pres_other , 1/10, -10010)  
    
    
    graph_5("Уступ: Omega(зел), F_dn(красн), Omega_пер(син), F_dn_пер(оранж) от t",
            dr_time, dr_pos, dr_pres, dr_pos_other, dr_pres_other, "4params-pick_stair2.png")
    
   
