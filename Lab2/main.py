import numpy as np
from scipy.linalg import lu, solve, norm


def createA(x):
    """ Creating a matrix """
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i][j] += j + 1
            if i == j and i != 0:
                A[i][j] = x + 1

    return A


def createE():
    """ Creating the identity matrix """
    E = np.zeros((N, N))
    for i in range(N):
        E[i][i] = 1

    return E


def invert(a):
    """ Inverting the matrix """
    P, L, U = lu(a)
    inverted = np.zeros((N, N))
    for i in range(N):
        e = np.zeros((N, 1))
        e[i] = 1
        x = solve(L, e)
        y = solve(U, x)
        for j in range(N):
            inverted[j][i] = y[j][0]

    return inverted


def condA(a):
    """ Computing the conditional number of matrix """
    return norm(a) * norm(invert(a))


def solution(x):
    """ Solution of lab. work """
    E = createE()
    A = createA(x)
    invA = invert(A)
    R = np.matmul(A, invA) - E
    normR = norm(R)
    print("Current x: ", x)
    print("Matrix A:\n", A)
    print("Inverted A:\n", invA)
    print("A's condition number: ", condA(A))
    print("Matrix R:\n", R)
    print("R's norm: ", normR)
    print("\n")


if __name__ == '__main__':
    N = 5

    solution(1.1)
    solution(1.001)
    solution(1.00001)
