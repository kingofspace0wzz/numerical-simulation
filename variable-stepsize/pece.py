import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy import optimize

'''
Approximated LTE for PECE with AB-2 and AM-2
LTE = (-1/12)/( -1/12 - 5/12 ) * (y_n - y_n(0))
    = 1/6 * (y_n - y_n(0))
'''

def pece1(a, b, y0, h, tol=10**(-3), frac=0.9):
    '''
    a: start point
    b: end point
    y0: initial condition
    h: initial stepsize
    tol: error tolerance
    frac: fraction of error tolerance

    Return:
        y: solutions
        t: mesh points
        f: derivatives
    '''

    # total number of points to be evaluated
    n = (int)((b-a)/h + 1)
    # a vector of points on which we evaluate the solution
    t = np.linspace(a,b,n)
    # create a vector of solutions y
    y = np.zeros((2, n))
    # set up initial conditions
    y[:, 0] = y0[:]
    # create a vector to store the values of f
    f = np.zeros((2, n))

    # use forward_euler method to evaluate y_1
    # first evaluate f
    f[0, 0] = -y[0, 0]
    f[1, 0] = -10 * y[1, 0]
    # then evaluate y_1 using midpoint method (2nd order RK) with f
    y[:, 1] = y[:, 0] + h/2 * f[:, 0]
    f_temp = f[:, 0]
    # f_temp is the derivative evaluated at the midpoint
    f_temp[0] = -y[0, 1]
    f_temp[1] = -10 * (y[1, 1] - np.square(h/2)) + 2*(h/2)
    y[:, 1] = y[:, 0] + h * f_temp

    for i in range(1, n-1):
        # y_0 and y_1 are evaluated, now we start from y_2
        # solve y1
        # predictor
        # WARNing: the new y and f may not reside on mesh points, because the stepsize may have been reduced
        y[:, i+1] = y[:, i] + h/2 * (3*f[:, i] - f[:, i-1])
        # update f using the value of y Approximated by predictor
        f[0, i+1] = -y[0, i+1]
        f[1, i+1] = -10 * (y[1, i+1] - np.square(t[i+1])) + 2 * t[i+1]

        # use functional iteration to solve corrector equation
        j = 0
        y_new = y[:, i+1]   # old predictor is the initial guess
        # old predictor
        old_predictor = y[:, i+1]
        while j <= 1000:
            # new corrector
            y_temp = y[:, i] + h/2 * (f[:, i] + f[:, i+1])
            # update f(t_i+1, y_i+1)
            f[0, i+1] = -y_temp[0]
            f[1, i+1] = -10 * (y_temp[1] - np.square(t[i+1])) + 2 * t[i+1]
            if la.norm(y_new - y_temp) <= 0.33*tol:
                # if condition of convergence satisfied

                print(1/6 * la.norm(old_predictor - y_temp))
                # control of local error by adapting stepsize
                if 1/6 * h * la.norm(old_predictor - y_temp) <= tol:
                    # accept this step
                    # set y_new to the new corrector y_temp
                    y_new = y_temp

                    break
                else:
                    # we have to reduce stepsize to control local error
                    # print(1/6*h*la.norm(old_predictor - y_temp, 2))
                    # print(1/6*h*min(abs(old_predictor - y_temp)))

                    # estimate of LTE
                    lte = 1/6*la.norm(old_predictor - y_temp)
                    r = frac * tol/(h*lte)
                    print(r)
                    print(h)
                    r = np.power(r, 1/3)

                    h = r * h
                    j = 0

                    # the stepsize has been reduced, now we use the new stepsize to compuate a new predictor y_new
                    old_predictor = y[:, i] + h/2 * (3*f[:, i] - f[:, i-1])  # y_new may not reside on any mesh points now

                    # update f, which may not be on mesh points neigher
                    f[0, i+1] = -old_predictor[0]
                    f[1, i+1] = -10 * (old_predictor[1] - np.square(t[i+1])) + 2 * t[i+1]

            else:
                j += 1
                # set y_temp to the previous corrector
                y_new = y_temp

            if j == 1000 and abs((y_temp - y_new).sum()) >= tol*len(y_temp):
                # if it does not converge
                # reduce stepsize
                h /= 2
                y_new = y_temp
                j = 0

        # after the while loop, we should have the values of y_new and f, but they are on points between i and i+1
        # we need second order interpolant to evaluate the solutions on mesh point i+1
        y[:, i+1] = y_new + (t[i+1] - t[i] - h) * f[:, i+1] + (f[:, i+1] - f[:, i])/h * np.square(t[i+1] - t[i] - h)

        # now update f using the new y
        f[0, i+1] = -y[0, i+1]
        f[1, i+1] = -10 * (y[1, i+1] - np.square(t[i+1])) + 2 * t[i+1]

    return y, t, f

def pece2(a, b, y0, h, tol=10**(-3), frac=0.9):
    '''
    a: start point
    b: end point
    y0: initial condition
    h: initial stepsize
    tol: error tolerance
    frac: fraction of error tolerance

    Return:
        y: solutions
        t: mesh points
        f: derivatives
    '''

    # total number of points to be evaluated
    n = (int)((b-a)/h + 1)
    # a vector of points on which we evaluate the solution
    t = np.linspace(a,b,n)
    # create a vector of solutions y
    y = np.zeros((2, n))
    # set up initial conditions
    y[:, 0] = y0[:]
    # create a vector to store the values of f
    f = np.zeros((2, n))

    # use forward_euler method to evaluate y_1
    # first evaluate f
    f[0, 0] = .25*y[0, 0] - .01*y[0, 0]*y[1, 0]
    f[1, 0] = -y[1, 0] + .01*y[0, 0]*y[1, 0]
    # then evaluate y_1 using midpoint method (2nd order RK) with f
    y[:, 1] = y[:, 0] + h/2 * f[:, 0]
    f_temp = f[:, 0]
    # f_temp is the derivative evaluated at the midpoint
    f_temp[0] = .25*y[0, 1] - .01*y[0, 0]*y[1, 1]
    f_temp[1] = -y[1, 1] + .01*y[0, 1]*y[1, 1]
    y[:, 1] = y[:, 0] + h * f_temp



    for i in range(1, n-1):
        # y_0 and y_1 are evaluated, now we start from y_2
        # solve y1
        # predictor
        # WARNing: the new y and f may not reside on mesh points, because the stepsize may have been reduced
        y[:, i+1] = y[:, i] + h/2 * (3*f[:, i] - f[:, i-2])
        # update f using the value of y Approximated by predictor
        f[0, i+1] = .25*y[0, i+1] - .01*y[0, i+1]*y[1, i+1]
        f[1, i+1] = -y[1, i+1] + .01*y[0, i+1]*y[1, i+1]

        # use functional iteration to solve corrector equation
        j = 0
        y_new = y[:, i+1]   # old predictor is the initial guess
        # old predictor
        old_predictor = y[:, i+1]
        while j <= 1000:
            # corrector equation
            y_temp = y[:, i] + h/2 * (f[:, i] + f[:, i+1])
            # update f(t_i+1, y_i+1)
            f[0, i+1] = .25*y_temp[0] - .01*y_temp[0]*y_temp[1]
            f[1, i+1] = -y_temp[1] + .01*y_temp[0]*y_temp[1]
            # print(la.norm(old_predictor - y_temp))
            if la.norm(y_new - y_temp) <= 5*tol:
                # if condition of convergence satisfied

                # control of local error by adapting stepsize
                if 1/6 * h * la.norm(old_predictor - y_temp) <= tol:
                    # accept this step
                    # set y_i+1 to y_temp
                    y_new = y_temp
                    break
                else:
                    # estimate of LTE
                    lte = 1/6*la.norm(old_predictor - y_temp)
                    r = frac * tol/(h*lte)

                    r = np.power(r, 1/3)

                    h = r * h
                    j = 0

                    # the stepsize has been reduced, now we use the new stepsize to compuate a new predictor y_new
                    old_predictor = y[:, i] + h/2 * (3*f[:, i] + f[:, i-1])  # y_new may not reside on any mesh points now

                    # update f, which may not be on mesh points neigher
                    f[0, i+1] = .25*old_predictor[0] - .01*old_predictor[0]*old_predictor[1]
                    f[1, i+1] = -old_predictor[1] + .01*old_predictor[0]*old_predictor[1]

            else:
                j += 1
                y_new = y_temp

            if j == 1000 and abs((y_temp - y_new).sum()) >= tol*len(y_temp):
                # if it does not converge
                # reduce stepsize
                h /= 2
                y_new = y_temp
                j = 0


        # after the while loop, we should have the values of y_new and f, but they are on points between i and i+1
        # we need second order interpolant to evaluate the solutions on mesh point i+1
        y[:, i+1] = y_new + (t[i+1] - t[i] - h) * f[:, i+1] + (f[:, i+1] - f[:, i])/h * np.square(t[i+1] - t[i] - h)

        # now update f using the new y
        f[0, i+1] = .25*y[0, i+1] - .01*y[0, i+1]*y[1, i+1]
        f[1, i+1] = -y[1, i+1] + .01*y[0, i+1]*y[1, i+1]


    return y, t, f

def pece3(a, b, y0, h, tol=10**(-3), frac=0.9):
    '''
    a: start point
    b: end point
    y0: initial condition
    h: initial stepsize
    tol: error tolerance
    frac: fraction of error tolerance

    Return:
        y: solutions
        t: mesh points
        f: derivatives
    '''

    # total number of points to be evaluated
    n = (int)((b-a)/h + 1)
    # a vector of points on which we evaluate the solution
    t = np.linspace(a,b,n)
    # create a vector of solutions y
    y = np.zeros((2, n))
    # set up initial conditions
    y[:, 0] = y0[:]
    # create a vector to store the values of f
    f = np.zeros((2, n))

    # use forward_euler method to evaluate y_1
    # first evaluate f
    f[0, 0] = y[1, 0]
    f[1, 0] = 2 * ( (1 - np.square(y[0, 0])) * y[1, 0] - y[0, 0] )
    # then evaluate y_1 using midpoint method (2nd order RK) with f
    y[:, 1] = y[:, 0] + h/2 * f[:, 0]
    f_temp = f[:, 0]
    # f_temp is the derivative evaluated at the midpoint
    f_temp[0] = y[1, 1]
    f_temp[1] = 2 * ( (1 - np.square(y[0, 1])) * y[1, 1] - y[0, 1] )
    y[:, 1] = y[:, 0] + h * f_temp

    for i in range(1, n-1):
        # y_0 and y_1 are evaluated, now we start from y_2
        # solve y1
        # predictor
        # WARNing: the new y and f may not reside on mesh points, because the stepsize may have been reduced
        y[:, i+1] = y[:, i] + h/2 * (3*f[:, i] - f[:, i-1])
        # update f using the value of y Approximated by predictor
        f[0, i+1] = y[1, i+1]
        f[1, i+1] = 2 * ( (1 - np.square(y[0, i+1])) * y[1, i+1] - y[0, i+1] )

        # use functional iteration to solve corrector equation
        j = 0
        y_new = y[:, i+1]   # old predictor is the initial guess
        # old predictor
        old_predictor = y[:, i+1]
        while j <= 1000:
            # new corrector
            y_temp = y[:, i] + h/2 * (f[:, i] + f[:, i+1])
            # update f(t_i+1, y_i+1)
            f[0, i+1] = y_temp[1]
            f[1, i+1] = 2 * ( (1 - np.square(y_temp[0])) * y_temp[1] - y_temp[0] )
            if la.norm(y_new - y_temp) <= 0.33*tol:
                # if condition of convergence satisfied

                print(1/6 * la.norm(old_predictor - y_temp))
                # control of local error by adapting stepsize
                if 1/6 * h * la.norm(old_predictor - y_temp) <= tol:
                    # accept this step
                    # set y_new to the new corrector y_temp
                    y_new = y_temp

                    break
                else:
                    # we have to reduce stepsize to control local error
                    # print(1/6*h*la.norm(old_predictor - y_temp, 2))
                    # print(1/6*h*min(abs(old_predictor - y_temp)))

                    # estimate of LTE
                    lte = 1/6*la.norm(old_predictor - y_temp)
                    r = frac * tol/(h*lte)
                    print(r)
                    print(h)
                    r = np.power(r, 1/3)

                    h = r * h
                    j = 0

                    # the stepsize has been reduced, now we use the new stepsize to compuate a new predictor y_new
                    old_predictor = y[:, i] + h/2 * (3*f[:, i] - f[:, i-1])  # y_new may not reside on any mesh points now

                    # update f, which may not be on mesh points neigher
                    f[0, i+1] = old_predictor[1]
                    f[1, i+1] = 2 * ( (1 - np.square(old_predictor[0])) * old_predictor[1] - old_predictor[0] )

            else:
                j += 1
                # set y_temp to the previous corrector
                y_new = y_temp

            if j == 1000 and abs((y_temp - y_new).sum()) >= tol*len(y_temp):
                # if it does not converge
                # reduce stepsize
                h /= 2
                y_new = y_temp
                j = 0

        # after the while loop, we should have the values of y_new and f, but they are on points between i and i+1
        # we need second order interpolant to evaluate the solutions on mesh point i+1
        y[:, i+1] = y_new + (t[i+1] - t[i] - h) * f[:, i+1] + (f[:, i+1] - f[:, i])/h * np.square(t[i+1] - t[i] - h)

        # now update f using the new y
        f[0, i+1] = y[1, i+1]
        f[1, i+1] = 2 * ( (1 - np.square(y[0, i+1])) * y[1, i+1] - y[0, i+1] )

    return y, t, f


def test1():
    y, t,_ = pece1(0, 1, [1,2], .01)

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(t, y[0])
    plt.subplot(2,1,2)
    plt.plot(t, y[1])
    plt.show()

def test2():
    y, t,_ = pece2(0, 100, [10, 10], .05)

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(t, y[0])
    plt.subplot(3,1,2)
    plt.plot(t, y[1])
    plt.subplot(3,1,3)
    plt.plot(y[0], y[1])
    plt.show()

def test3():
    y, t,_ = pece3(0, 11, [2, 0], .01)

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(t, y[0])
    plt.subplot(2,1,2)
    plt.plot(t, y[1])
    plt.show()

if __name__ == '__main__':
    test3()
