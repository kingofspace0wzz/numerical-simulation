import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def forward_euler(a, b, y0, h):
    '''
    Forward euler method  (theta = 1)

    Args:
      a: start point
      b: end point
      y0: initial condition
      h: step size
    '''

    n = (int)((b-a)/h + 1) # number of grid points
    t = np.linspace(a, b, n)
    y = np.zeros((2, n))

    y[:, 0] = y0[:]  # initial condition; y0 is a vector
    for i in range(n-1):
        y[0, i+1] = y[0, i] + h*(-y[0, i])
        y[1, i+1] = y[1, i]+ h*(-100*(y[1, i] - np.sin(t[i])) + np.cos(t[i]))

    return t, y


def implicit_euler(a, b, y0, h):
    '''
    Implicit euler method  (theta = 0.5)

    Args:
      a: start point
      b: end point
      y0: initial condition
      h: step size
    '''
    n = (int)((b-a)/h + 1)  # number of grid points
    t = np.linspace(a, b, n)

    y = np.zeros((2, n))
    y[0, 0], y[1, 0] = y0[0], y0[1]     # initial conditions

    for i in range(n-1):
        y_temp = y[:, i]  # initial guess of y_n+1^(k=0) is the current y_n

        for k in range(1000):
            A = np.eye(2) - h*np.array([[-1, 0], [0, -100]])    # Identity matrix - h * Jacobian Matrix
            b = np.dot(A, y_temp) - y_temp + y[:, i] + h*np.array(
                [ -y_temp[0], -100*(y_temp[1] - np.sin(t[i+1])) + np.cos(t[i+1]) ])

            y[:, i+1] = np.linalg.solve(A, b)   # solve this linear equation to get y_(n+1)

            if (abs(y[0, i+1] - y_temp[0]) < 0.001 and abs(y[1, i+1] - y_temp[1]) < 0.001):     # stopping criteria
                break
            y_temp = y[:, i+1]

    return t, y


def trapezoidal(a, b, y0, h):
    '''
    Trapezoidal method  (theta = 0.5)

    Args:
      a: start point
      b: end point
      y0: initial condition
      h: step size
    '''
    n = (int)((b-a)/h + 1)  # number of grid points
    t = np.linspace(a, b, n)

    y = np.zeros((2, n))
    y[0, 0], y[1, 0] = y0[0], y0[1]     # initial condition

    for i in range(n-1):
        y_temp = y[:, i]  # initial guess of y_n+1^(k=0) is the current y_n

        for k in range(1000):
            A = np.eye(2) - h*np.array([[-1, 0], [0, -100]])    # Identity matrix - h * Jacobian Matrix
            b = np.dot(A, y_temp) - y_temp + y[:, i] + 0.5*h*np.array(
                [-y_temp[0], -100*(y_temp[1] - np.sin(t[i+1])) + np.cos(t[i+1])]) + 0.5*h*np.array([-y[0, i], -100*(y[1, i] - np.sin(t[i])) + np.cos(t[i])])

            y[:, i+1] = np.linalg.solve(A, b)   # solve this linear equation to get y_(n+1)

            # stopping criteria
            if (abs(y[0, i+1] - y_temp[0]) < 0.001 and abs(y[1, i+1] - y_temp[1]) < 0.001):
                break
            y_temp = y[:, i+1]

    return t, y



def test():

    # h = 0.01
    t, y = forward_euler(0, 1, [1,2], 0.01)

    plt.figure(1)
    plt.subplot(331)
    plt.plot(t, y[0])
    plt.ylabel("y1")
    plt.title("Forward-Euler, h=0.01")
    plt.subplot(334)
    plt.plot(t, y[1])
    plt.ylabel("y2")
    plt.subplot(337)
    plt.plot(y[0], y[1])
    plt.ylabel("y1 vs y2")
    plt.xlabel("t")

    t, y = trapezoidal(0, 1, [1, 2], 0.01)

    plt.subplot(332)
    plt.plot(t, y[0])
    plt.title("Trapezoidal, h=0.01")
    plt.subplot(335)
    plt.plot(t, y[1])
    plt.subplot(338)
    plt.plot(y[0], y[1])
    plt.xlabel("t")

    t, y = implicit_euler(0, 1, [1, 2], 0.01)

    plt.subplot(333)
    plt.plot(t, y[0])
    plt.title("Implicit-Euler, h=0.01")
    plt.subplot(336)
    plt.plot(t, y[1])
    plt.subplot(339)
    plt.plot(y[0], y[1])
    plt.ylabel("y1 vs y2")
    plt.xlabel("t")

    # h = 0.05
    t, y = forward_euler(0, 1, [1, 2], 0.05)

    plt.figure(2)
    plt.subplot(331)
    plt.plot(t, y[0])
    plt.ylabel("y1")
    plt.title("Forward-Euler, h=0.05")
    plt.subplot(334)
    plt.plot(t, y[1])
    plt.ylabel("y2")
    plt.subplot(337)
    plt.plot(y[0], y[1])
    plt.ylabel("y1 vs y2")
    plt.xlabel("t")

    t, y = trapezoidal(0, 1, [1,2], 0.05)

    plt.subplot(332)
    plt.plot(t, y[0])
    plt.title("Trapezoidal, h=0.05")
    plt.subplot(335)
    plt.plot(t, y[1])
    plt.subplot(338)
    plt.plot(y[0], y[1])
    plt.xlabel("t")

    t, y = implicit_euler(0, 1, [1, 2], 0.05)

    plt.subplot(333)
    plt.plot(t, y[0])
    plt.title("Implicit-Euler, h=0.05")
    plt.subplot(336)
    plt.plot(t, y[1])
    plt.subplot(339)
    plt.plot(y[0], y[1])
    plt.xlabel("t")

    plt.show()


if __name__ == '__main__':
    test()
