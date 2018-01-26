import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def forward_euler(f, a, b, y0, h):
    '''
    Forward euler method

    Args:
      f: f = y'
    '''

    n = (int)((b-a)/h + 1)
    t = np.linspace(a,b,n)
    y = np.zeros((len(f),n))

    y[:, 0] = y0[:] # initial condition; y0 is a vector
    for i in range(n-1):
        y[:, i+1] = y[:, i] + [h*f[j](t[i], y[j,i]) for j in range(len(f))]

    return t, y

def implicit_euler(f, a, b, y0, h):
    '''
    Implicit euler method

    Args:

    '''
    n = (int)((b-a)/h + 1)
    t = np.linspace(a,b,n)
    y = np.zeros((len(f),n))

    y[:, 0] = y0[:] # initial condition; y0 is a vector

    # for i in range(n-1):
    #     for j in range(len(f)):
    #         y[j,i] = optimize.newton(lambda y_next: h*f[j](t[i+1], y_next) - y_next + y[j,i],
    #                                  y[j,i])

    for i in range(n-1):

        y[:, i] = [optimize.newton(lambda y_next: h*f[j](t[i+1], y_next) - y_next + y[j,i], y[j,i])
                  for j in range(len(f))]

    return t, y

def newton(f, df, x0, tol = 0.001, iter=1000):
    '''
    Newton Raphson method

    Args:
      f: polynomial function of x
      df: derivative as a function of x
      x0: initial state
      tol: tolerance; dault tol = 0.001
      iter: maximum number of iterations; dault iter = 1000

     return:
       root, if there is one
       False, if there is no root
    '''

    for i in range(iter):
        x1 = x0 - (f(x0)/df(x0))
        if abs(x1 - x0) < tol:
            return x1
        else:
            x0 = x1

    return False


#----------test-----------

def test_newton():
    root = newton(lambda x: x**3 - 1, lambda x: 3*x**2, 0.5 )
    print(root)

def test2():
    t1, y1 = forward_euler([lambda t, y2: -100*(y2 - np.sin(t)) + np.cos(t),
                            lambda t, y1: -y1],
                            0, 1, [2, 1], 0.01)

    t2, y2 = forward_euler([lambda t, y2: -100*(y2 - np.sin(t)) + np.cos(t),
                            lambda t, y1: -y1],
                            0, 1, [2, 1], 0.05)

    plt.figure(1)

    plt.subplot(321)
    plt.plot(t1 ,y1[0])
    plt.xlabel('t')
    plt.ylabel('y2(t)')
    plt.title('step size h = 0.01')

    plt.subplot(322)
    plt.plot(t2,y2[0])
    plt.xlabel('t')
    plt.ylabel('y2(t)')
    plt.title('step size h = 0.05')


    plt.subplot(323)
    plt.plot(t1, y1[1])
    plt.xlabel('t')
    plt.ylabel('y1(t)')

    plt.subplot(324)
    plt.plot(t2 ,y2[1])
    plt.xlabel('t')
    plt.ylabel('y1(t)')

    plt.subplot(325)
    plt.plot(y1[1], y1[0])
    plt.xlabel('y1(t)')
    plt.ylabel('y2(t)')

    plt.subplot(326)
    plt.plot(y2[1] ,y2[0])
    plt.xlabel('y1(t)')
    plt.ylabel('y2(t)')


    plt.figure(2)
    x = np.linspace(0, 1, 1/0.01)
    y = np.sin(x)
    plt.ylim(0,2)
    plt.plot(x, y)
    plt.xlabel('t')
    plt.ylabel('y=sin(t)')
    plt.title('y = sin(t)')

    plt.show()

def test3():
    t1, y1 = implicit_euler([lambda t, y2: -100*(y2 - np.sin(t)) + np.cos(t),
                            lambda t, y1: -y1],
                            0, 1, [2, 1], 0.05)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(t1, y1[0])
    plt.subplot(212)
    plt.plot(t1, y1[1])

    plt.show()

if __name__ == '__main__':
    test3()
