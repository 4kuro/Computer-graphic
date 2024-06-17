import numpy as np
import matplotlib.pyplot as plt
def moves(axiom, rules: dict, max_iter):
    for _ in range(max_iter):
        newaxiom = ""
        for el in axiom:
            if el in rules.keys():
                newaxiom += str(rules[el])
            else:
                newaxiom += el
        axiom = newaxiom

    return axiom
def show_fractal(axiom, rules, max_iter, fi, dfi):
    if isinstance(dfi, int):
        dfi = np.radians(dfi)
    fractal = moves(axiom, rules, max_iter)
    N=len(fractal)
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    L = 2
    for i in range(N):
        x[i+1]=x[i]
        y[i+1]=y[i]
        if fractal[i] == "F":
            x[i+1] += L*np.cos(fi)
            y[i+1] += L*np.sin(fi)
        elif fractal[i]=="+":
            fi+=dfi
        elif fractal[i]=="-":
            fi-=dfi
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y, linewidth=0.4)
    plt.show()
#сніжинка
axiom1 = "F++F++F"
rules1 = {"F":"F-F++F-F"}
max_iter1 = 5
fi1=0
dfi1=np.pi/3
show_fractal(axiom1, rules1, max_iter1, fi1, dfi1)
#Хрестоподібний фрактал
axiom2 = "F+XF+F+XF"
rules2 = {"X":"XF-F+F-XF+F+XF-F+F-X"}
max_iter2 = 5
fi2 = 0
dfi2 = np.pi/2
show_fractal(axiom2, rules2, max_iter2, fi2, dfi2)
#Квадратний острівець Коха
axiom3 = "F+F+F+F"
rules3 = {"F":"F+F-F-FFF+F+F-F"}
max_iter3 = 5
fi3=0
dfi3=np.pi/2
show_fractal(axiom3, rules3, max_iter3, fi3, dfi3)
#Наконечник стріли Серпінського
axiom4 = "YF"
rules4 = {"X":"YF+XF+Y", "Y":"XF-YF-X"}
max_iter4 = 8
fi4=0
dfi4=np.pi/3
show_fractal(axiom4, rules4, max_iter4, fi4, dfi4)
#Крива дракона
axiom5 = "FX"
rules5 ={"X":"X+YF+", "Y":"-FX-Y"}
max_iter5 = 20
fi5=0
dfi5=90
show_fractal(axiom5, rules5, max_iter5, fi5, dfi5)
#Крива Леві
axiom6 = "FX"
rules6 = {"F":"-F++F-"}
max_iter6 = 15
fi6=0
dfi6=np.pi/4
show_fractal(axiom6, rules6, max_iter6, fi6, dfi6)
#Спробувати змінити кут θ (angle) при побудові цих об’єктів. Отримати новий фрактальний об’єкт.
dfi6_4 = np.pi/7
show_fractal(axiom6, rules6, max_iter6, fi6, dfi6_4)
#Фрактальний басейн
axiom7 = "-D--D"
rules7 = {"A":"F++FFFF--F--FFFF++F++FFFF--F",
         "B":"F--FFFF++F++FFFF--F--FFFF++F",
         "C":"BFA--BFA",
         "D": "CFC--CFC"}
max_iter7 = 5
fi7=0
dfi7=np.pi/4
show_fractal(axiom7, rules7, max_iter7, fi7, dfi7)
#Фрактальна плитка
axiom8 = "F+F+F+F"
rules8 ={"F":"FF+F-F+F+FF"}
max_iter8 = 5
fi8=0
dfi8=np.pi/2
show_fractal(axiom8, rules8, max_iter8, fi8, dfi8)
