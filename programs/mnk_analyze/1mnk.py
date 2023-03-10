import matplotlib.pyplot as plt
import numpy as np
from pylab import *

def calibrate(list_, koef_d, koef_p):
    l = []
    for i in list_:
        l.append(i/koef_d + koef_p)
    return l
    
if __name__ == "__main__":    
    time, pos_r, pos_l, pres_up_r, pres_up_l, pres_dn_r, pres_dn_l, trq_r, trq_l = [], [], [], [], [], [], [], [], []
    f = open('1_1.txt','r')
    try:
    #работа с файлом
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
    finally:
        f.close()

    #делаем выборку до 150 строк таблицы
    pos_r_short, pdn_r_short = pos_r[0:150], pres_dn_r[0:150]
    
    pres_up_l = calibrate(pres_up_l, 1/10, -10100) 
    pres_dn_l = calibrate(pres_dn_l , 1/10, -10010)
    
    fig, ax = plt.subplots()
    #точечный график зависимости правых нижних от правых верхних давлений
    ax.scatter(pres_up_r, pres_dn_r, c="red", label="Правый, pres_up от pres_dn")
    #заголовок для графика
    ax.set_title('Правый, pres_up от pres_dn') 

    fig.set_figwidth(8)     #  ширина 
    fig.set_figheight(8)    #  высота
    plt.savefig('pres_up_dn_right.png')
    
    figure()
    fig = plt.figure()
    ax = fig.add_subplot(111, label="scat")
    ax2 =fig.add_subplot(111, label="line1", frame_on=False)
    ax3 =fig.add_subplot(111, label="line2", frame_on=False)
    #точечный график зависимости левых нижних от левых верхних давлений
    ax.scatter(pres_up_l, pres_dn_l, color="red", label="Левый, F_up от F_dn")
    #размер списка
    N=len(pres_up_l)
    xx = np.linspace(1000, 12000, N)
    #предсказуемый график зависимости у=х синий
    yy=xx
    ax2.plot(xx, yy, color="C0")
    ax2.set_xticks([])
    ax2.set_yticks([])
    #поиск средних значений
    mx = np.array(pres_up_l).sum()/N
    my = np.array(pres_dn_l).sum()/N
    #скалярные произведения транспонированного вектора на обычный
    a_2 = np.dot(xx.T, xx)/N
    a_11 = np.dot(xx.T, yy)/N
    #искомые коэффициенты
    kk = (a_11 - mx*my)/(a_2 - mx**2)
    bb = my - kk*mx
    print('k_approx =',kk,'b_approx =',bb)
    #полученный график зависимости у=х зеленый
    #если предположение верно, зеленый будет лежать очень близко к синему
    ff = np.array([kk*z+bb for z in range(N)])
    ax3.plot(xx, ff, color="C2")
    ax3.set_xticks([])
    ax3.set_yticks([])   
    #заголовок для графика
    ax.set_title('Левый, F_up от F_dn')    

    fig.set_figwidth(8)     #  ширина 
    fig.set_figheight(8)    #  высота
    plt.savefig('pres_up_dn_left.png')
    
