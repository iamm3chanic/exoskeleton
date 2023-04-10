import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from numpy.fft import rfft, irfft
from numpy.random import uniform


def FH(x):  # ступенчатая функция Хэвисайда 
         if x>=0:
                  q=1
         else:
                  q=0
         return q

def graph_1(name, l1, l2):
    plt.title(name) # заголовок
    plt.xlabel("x")         # ось абсцисс
    plt.ylabel("y")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(l1,l2)  # построение графика
    plt.show()    

def graph_2(title, l1, l2, filename):
    plt.title(title) # заголовок
    plt.xlabel("x")         # ось абсцисс
    plt.ylabel("y")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(l1,l2)  # построение графика
    plt.savefig(filename)
    figure()

def graph_3(title, time, l1, l2, l3, l4, filename):
    plt.title(title) # заголовок
    plt.xlabel("x")         # ось абсцисс
    plt.ylabel("y")    # ось ординат
    plt.grid()              # включение отображение сетки
    plt.plot(time,l1)  
    plt.plot(time,l2)
    plt.plot(time,l3)
    plt.plot(time,l4)
    plt.savefig(filename)
    figure()
    
##############################################
    
if __name__ == "__main__":    
    time, pos_r, pos_l, pres_up_r, pres_up_l, pres_dn_r, pres_dn_l, trq_r, trq_l = [], [], [], [], [], [], [], [], []
    f = open('1.txt','r')
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
    finally:
        f.close()

    
    # разложение в ряд фурье
    k=np.array(time) #289 итераций
    k = np.delete(k, 288)
    T=np.pi;z=T/288; m=[t*z for t in k] #задание для дискретизации функции на 289 отсчётов
    v=np.array(pos_r) #дискретизация исходной функции, с шумом
    v = np.delete(v, 288)
    # для начала исследуем позицию правой ноги
    #vs= [f(t)+np.random.uniform(0,0.5) for t in m]# добавление шума
    plt.figure()
    plt.title("Фильтрация аналоговых сигналов  \n Окно исходной и зашумленной функций")
    #plt.plot(k,v, label='Окно исходной функции шириной pi')
    plt.plot(k,v,label='Окно зашумленной функции шириной pi')
    plt.xlabel("Отсчёты -k")
    plt.ylabel("Амплитуда А")
    plt.legend(loc='best')
    plt.grid(True)
    al=2# степень фильтрации высших гармоник
    fs=np. fft.rfft(v)# переход из временной области в частотную с помощью БПФ
    g=[fs[j]*FH(abs(fs[j])-2) for j in np.arange(0,145,1)]# фильтрация высших гармоник
    h=np.fft.irfft(g) # возврат во временную область
    plt.figure()
    plt.title("Фильтрация аналоговых сигналов  \n Результат фильтрации")
    plt.plot(k,v,label='Окно исходной функции шириной pi')
    plt.plot(k,h, label='Окно результата фильтрации шириной pi')
    plt.xlabel("Отсчёты -k")
    plt.ylabel("Амплитуда А")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('furye.png')
    
