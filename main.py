from ode import forward_euler
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Numerical solutions of y1 & y2 with step size h=0.01
    t1, y1 = forward_euler([lambda t, y2: -100*(y2 - np.sin(t)) + np.cos(t),
                            lambda t, y1: -y1],
                            0, 1, [2, 1], 0.01)

    # Numercial solutions of y1 & y2 with step size h=0.05
    t2, y2 = forward_euler([lambda t, y2: -100*(y2 - np.sin(t)) + np.cos(t),
                            lambda t, y1: -y1],
                            0, 1, [2, 1], 0.05)

    # plot the solution curves
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
