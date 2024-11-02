import numpy as np

from scipy import integrate
from scipy.optimize import fsolve
from prettytable import PrettyTable
from matplotlib import pyplot


def ans(t):
    return t ** 3


def rk45(f, T, Y0):
    r = integrate.ode(f).set_integrator('dopri5', atol=1e-6)
    r.set_initial_value(Y0, T[0])

    Y = np.zeros((len(T), len(Y0)))
    Y[0] = Y0

    for i in range(1, len(T)):
        Y[i] = r.integrate(T[i])

    return Y


def trapezoidal(f, T, Y0):
    Y = np.zeros((len(T), len(Y0)))
    Y[0] = Y0
    for i in range(1, len(T)):
        f1 = lambda Y: f(T[i], Y)

        def equations(X):
            return [
                Y[i-1][j] + 0.5*(T[i] - T[i-1])*(f1(X)[j] + f1(Y[i-1])[j]) - X[j] for j in range(len(X))
            ]

        root = fsolve(equations, [0] * len(Y0), xtol=1e-14, maxfev=2 ** 30)
        for j in range(len(Y[i])):
            Y[i][j] = Y[i-1][j] + 0.5*(T[i] - T[i-1])*(f1(root)[j] + f1(Y[i-1])[j])

    return Y


def f(t, Y):
    dY = np.zeros(len(Y))
    dY[0] = Y[1]
    dY[1] = (6 * Y[0]) / (t ** 2)
    return dY


if __name__ == "__main__":
    H = [0.1, 0.05, 0.025, 0.0125]
    rk45_global_err = []
    tr_global_err = []
    rk45_local_err = []
    tr_local_err = []
    for h in H:
        rng = np.arange(1, 2 + h - 1e-9, h)
        rng = np.round(rng, decimals=2)
        n_steps = len(rng)
        Y0 = [1, 3]
        res_rk45 = [i[0] for i in rk45(f, rng, Y0)]
        res_tr = [i[0] for i in trapezoidal(f, rng, Y0)]
        res_precise = [ans(rng[i]) for i in range(n_steps)]
        if h == 0.1:
            table = PrettyTable()
            table.add_column("t", rng)
            table.add_column("rk45", np.round(res_rk45, decimals=10))
            table.add_column("trapezoidal", np.round(res_tr, decimals=10))
            table.add_column("precise", np.round(res_precise, decimals=10))
            print("Data for h = 0.1:")
            print(table)
        else:
            table = PrettyTable()
            table.add_column("t", rng[:4])
            table.add_column("rk45", np.round(res_rk45[:4], decimals=10))
            table.add_column("trapezoidal", np.round(res_tr[:4], decimals=10))
            table.add_column("precise", np.round(res_precise[:4], decimals=10))
            print(f'Data for h = {h}')
            print(table)
        rk45_global_err.append(sum([abs(res_rk45[i] - res_precise[i]) for i in range(n_steps)]) / n_steps)
        tr_global_err.append(sum([abs(res_tr[i] - res_precise[i]) for i in range(n_steps)]) / n_steps)
        rk45_local_err.append(abs(res_rk45[1] - res_precise[1]))
        tr_local_err.append(abs(res_tr[1] - res_precise[1]))
    table_errs = PrettyTable()
    print("Local errors")
    table_errs.add_column("h", H)
    table_errs.add_column("rk45", rk45_local_err)
    table_errs.add_column("tr", tr_local_err)
    print(table_errs)
    table_global_errs = PrettyTable()
    print("Global errors")
    table_global_errs.add_column("h", H)
    table_global_errs.add_column("rk45", rk45_global_err)
    table_global_errs.add_column("tr", tr_global_err)
    print(table_global_errs)
    pyplot.title("rk45_err_rel")
    pyplot.plot(H, rk45_local_err, color="Blue")
    pyplot.figure()
    pyplot.title("tr_err_rel")
    pyplot.plot(H, tr_local_err, color="Red")
    pyplot.show()
