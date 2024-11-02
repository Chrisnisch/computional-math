from scipy.integrate import quad
from scipy.interpolate import lagrange, CubicSpline
from numpy import arange, e
from prettytable import PrettyTable
from matplotlib import pyplot


def func(x: float):
    def u(z: float):
        return 1 / (e ** z * (z + x))

    return u


def integral(f):
    return quad(f, 0, 20, limit=30)


def print_values_table():
    table = PrettyTable()
    table.add_column("x", x_i)
    table.add_column("f(x)", G_i)
    table.add_column("L(x)", values_l)
    table.add_column("S(x)", values_s)
    print(table)


def print_errs_table():
    table = PrettyTable()
    table.add_column("x", x_i)
    table.add_column("errL", errs_l)
    table.add_column("errS", errs_s)
    print(table)


def draw_function_plots():
    pyplot.title("Function")
    pyplot.plot(x_i, G_i, color="Red")

    pyplot.figure()
    pyplot.title("Lagrange")
    pyplot.plot(x_i, values_l, color="Blue")

    pyplot.figure()
    pyplot.title("Spline")
    pyplot.plot(x_i, values_s, color="Green")

    pyplot.show()


def draw_errs_plots():
    pyplot.title("Errors")
    pyplot.plot(x_i, errs_l, label="e_l(x)", color="Blue")
    pyplot.legend()
    pyplot.plot(x_i, errs_s, label="e_s(x)", color="Green")
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":

    a = 1
    b = 4
    h = 0.375

    # Задание узлов
    x_k = arange(a, b + h, h)

    # Задание функции внутри интеграла
    inner_values = []
    for x_h in x_k:
        inner_values.append(func(x_h))

    # Вычисление значения функции в узлах с помощью аналога quanc8
    G_k = []
    for f in inner_values:
        G_k.append(integral(f)[0])

    # Построение полинома Лагранжа
    poly = lagrange(x_k, G_k)

    # Построене сплайн-функции
    spline = CubicSpline(x_k, G_k)

    # Промежуток для сравнения аппроксимаций
    x_i = 1.1875 + 0.375 * arange(8)

    # Вычисление значений полинома Лагранжа в точках x_i
    values_l = poly(x_i)
    # Вычислений значений сплайна в точках x_i
    values_s = spline(x_i)

    inner_values2 = []
    for x_h in x_i:
        inner_values2.append(func(x_h))

    # Вычисление значения функции в точках x_i
    G_i = []
    for f in inner_values2:
        G_i.append(integral(f)[0])

    # Вывод таблицы значений
    print_values_table()

    # Вычисление погрешностей для значений полинома Лагранжа и сплайна
    errs_l = [abs(x - y) for x, y in zip(G_i, values_l)]
    errs_s = [abs(x - y) for x, y in zip(G_i, values_s)]

    # Вывод таблицы погрешностей
    print_errs_table()

    # Построение графиков функции и графиков погрешностей
    draw_function_plots()
    draw_errs_plots()

    print(poly)
    print(spline)