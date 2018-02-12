import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def forward_euler(a, b, y0, h):
    '''
    Forward euler method (theta = 1)

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
        y[0, i+1] = y[0, i] + h*(0.25*y[0, i] - 0.01*y[0, i]*y[1, i])
        y[1, i+1] = y[1, i] + h*(-y[1, i] + 0.01*y[0, i]*y[1, i])

    return t, y

def implicit_euler(a, b, y0, h):
    '''
    Implicit euler method  (theta = 0)

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
            A = np.eye(2) - h*np.array([[0.25 - 0.01*y_temp[1], 0.01*y_temp[0]], [0.01*y_temp[1], -1+0.01*y_temp[0]]])  # identity matrix - h * Jacobian Matrix
            b = np.dot(A, y_temp) - y_temp + y[:, i] + h*np.array( [0.25*y_temp[0] - 0.01*y_temp[0]*y_temp[1], -y_temp[1] + 0.01*y_temp[0]*y_temp[1]] ) 
            
            y[:, i+1] = np.linalg.solve(A, b)   # solve this linear equations to get y_(n+1)
            
            if (abs(y[0, i+1] - y_temp[0]) < 0.001 and abs(y[1, i+1] - y_temp[1]) < 0.001):     # stoppoint criteria
                break
            y_temp = y[:, i+1]

    return t, y

def trapezoidal(a, b, y0, h):
    '''
    trapezoidal method  (theta = 0.5)

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
            A = np.eye(2) - h*np.array([[0.25 - 0.01*y_temp[1],
                                         0.01*y_temp[0]], [0.01*y_temp[1], -1+0.01*y_temp[0]]])
            b = np.dot(A, y_temp) - y_temp + y[:, i] + 0.5*h*np.array(
                [0.25*y_temp[0] - 0.01*y_temp[0]*y_temp[1], -y_temp[1] + 0.01*y_temp[0]*y_temp[1]]) + 0.5*h*np.array([0.25*y[0, i] - 0.01*y[0, i]*y[1, i], -y[1, i] + 0.01*y[0, i]*y[1, i]])

            y[:, i+1] = np.linalg.solve(A, b)

            if (abs(y[0, i+1] - y_temp[0]) < 0.001 and abs(y[1, i+1] - y_temp[1]) < 0.001):     # stopping criteria
                break
            y_temp = y[:, i+1]

    return t, y

def test():

    t, y = forward_euler(0, 100, [10, 10], 0.001)
    
    # h = 0.001
    plt.figure(1)
    plt.subplot(331)
    plt.plot(t, y[0])
    plt.ylabel("y1")
    plt.title("Forward-Euler, h=0.001")
    plt.subplot(334)
    plt.plot(t, y[1])
    plt.ylabel("y2")
    plt.subplot(337)
    plt.plot(y[0], y[1])
    plt.ylabel("y1 vs y2")
    plt.xlabel("t")

    t, y = trapezoidal(0, 100, [10, 10], 0.001)

    plt.subplot(332)
    plt.plot(t, y[0])
    plt.title("Trapezoidal, h=0.001")
    plt.subplot(335)
    plt.plot(t, y[1])
    plt.subplot(338)
    plt.plot(y[0], y[1])
    plt.xlabel("t")

    t, y = implicit_euler(0, 100, [10, 10], 0.001) 

    plt.subplot(333)
    plt.plot(t, y[0])
    plt.title("Implicit-Euler, h=0.001")
    plt.subplot(336)
    plt.plot(t, y[1])
    plt.subplot(339)
    plt.plot(y[0], y[1])
    plt.ylabel("y1 vs y2")
    plt.xlabel("t")


    # h = 0.1
    t, y = forward_euler(0, 100, [10, 10], 0.1)

    plt.figure(2)
    plt.subplot(331)
    plt.plot(t, y[0])
    plt.ylabel("y1")
    plt.title("Forward-Euler, h=0.1")
    plt.subplot(334)
    plt.plot(t, y[1])
    plt.ylabel("y2")
    plt.subplot(337)
    plt.plot(y[0], y[1])
    plt.ylabel("y1 vs y2")
    plt.xlabel("t")

    t, y = trapezoidal(0, 100, [10, 10], 0.1)

    plt.subplot(332)
    plt.plot(t, y[0])
    plt.title("Trapezoidal, h=0.1")
    plt.subplot(335)
    plt.plot(t, y[1])
    plt.subplot(338)
    plt.plot(y[0], y[1])
    plt.xlabel("t")

    t, y = implicit_euler(0, 100, [10, 10], 0.1)

    plt.subplot(333)
    plt.plot(t, y[0])
    plt.title("Implicit-Euler, h=0.1")
    plt.subplot(336)
    plt.plot(t, y[1])
    plt.subplot(339)
    plt.plot(y[0], y[1])
    plt.xlabel("t")

    plt.show()

if __name__ == '__main__':
    
    test()

