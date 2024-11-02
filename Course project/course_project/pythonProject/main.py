import scipy
import numpy as np
from prettytable import PrettyTable
from matplotlib import pyplot

""" Решение СЛАУ """
matrix_A = np.array([[10, 1, 4, 0], [1, 10, 5, -1], [4, 5, 10, 7], [0, -1, 7, 9]])
vector_b = [5, 13, 29, 24]
lu, piv = scipy.linalg.lu_factor(matrix_A)
x = scipy.linalg.lu_solve((lu, piv), vector_b)
x_names = 'ABFG'
print(f'Число обусловленности матрицы А системы (2):\ncond(A) = {np.linalg.cond(matrix_A)}')
print('Решение системы (2):')
for i in range(len(x)):
    print(f'{x_names[i]} = {x[i]}')

""" Вычисление интеграла """
integral = scipy.integrate.quad(lambda x1: 1 / (1 + x1 ** 2) ** (1 / 3), 0.2, 0.3)
l = 10.20638 * integral[0]
print(f'Значение l = {l}, погрешность интегрирования ε = {integral[1]}')

""" Решение ДУ """
A, B, F, G = x[0], x[1], x[2], x[3]


def func(x, Z):
    dZ = np.zeros(len(Z))
    dZ[0] = Z[1]
    dZ[1] = x * (Z[0] - dZ[0])
    return dZ


def rkf45(f, T, Y0):
    r = scipy.integrate.ode(f).set_integrator('dopri5', atol=1e-6)
    r.set_initial_value(Y0, T[0])

    Y = np.zeros((len(T), len(Y0)))
    Y[0] = Y0
    for i in range(1, len(T)):
        Y[i] = r.integrate(T[i])

    return Y


def solve(T, Y0):
    start_conditions = Y0
    res_rkf45 = [i[0] for i in rkf45(func, T, start_conditions)]
    return res_rkf45


def T_func(x, t, y):
    return scipy.interpolate.CubicSpline(t, y)(x)


def L_ua(ua):
    res = solve(rng, [A, ua])
    return T_func(l, rng, res) - B


rng = np.linspace(0, l, 400)

res = solve(rng, [A, F])
u_a = T_func(l, rng, res)
print(f'Выберем T`(0) = F, тогда T(l): {u_a}')
print(f'"Недолёт"')

res = solve(rng, [A, G])
u_b = T_func(l, rng, res)
print(f'Выберем T`(0) = G, тогда T(l): {u_b}')
print(f'"Перелёт"')

root = scipy.optimize.brentq(L_ua, u_a, u_b)
print(f'Корень уравнения найденный с помощью метода Брента: u_a = {root}')
res = solve(rng, [A, root])
print(f'T(l) = {T_func(l, rng, res)}')

table = PrettyTable()
for_table = []
for i in range(0, len(rng), 40):
    for_table.append(rng[i])
for_table.append(rng[-1])
table.add_column('t', for_table)

for_table = []
for i in range(0, len(res), 40):
    for_table.append(res[i])
for_table.append(res[-1])
table.add_column('T(t) (rkf45)', for_table)

print(table)

pyplot.title('T(x)')
pyplot.plot(rng, res)
pyplot.show()

""" Мутация начальных условий """

pyplot.title('Мутация начальных условий T(x)')

pyplot.plot(rng, res, label='0')

res = solve(rng, [A, root + 0.1])
pyplot.plot(rng, res, 'r', label='+0.1')

res = solve(rng, [A, root - 0.1])
pyplot.plot(rng, res, 'r--', label='-0.1')

res = solve(rng, [A, root + 0.2])
pyplot.plot(rng, res, 'g', label='+0.2')

res = solve(rng, [A, root - 0.2])
pyplot.plot(rng, res, 'g--', label='-0.2')

res = solve(rng, [A, root + 0.3])
pyplot.plot(rng, res, 'y', label='+0.3')

res = solve(rng, [A, root - 0.3])
pyplot.plot(rng, res, 'y--', label='-0.3')

pyplot.legend()
pyplot.show()
