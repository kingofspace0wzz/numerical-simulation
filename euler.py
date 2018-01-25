import numpy as np
import matplotlib.pyplot as plt

class ode:
    ''' Ode Solver'''
    def __init__(self, f):

        self.f = f # y' = f(t, y)
        self.t = []
        self.y = []
    def forward_euler(self, a, b, y0, h):

        n = (int)((b-a)/h + 1)
        t = np.linspace(a,b,n)
        y = np.zeros(n)
        y[0] = y0 # initial condition
        for i in range(n-1):
            y[i+1] = y[i] + h*self.f(t[i], y[i])

        self.t =t
        self.y =y
        return t, y

    def plot(self):
        '''Plot the solution curve of your ode'''
        plt.plot(self.t, self.y)
        plt.xlabel("t")
        plt.ylabel('y(t)')
        plt.show()


def forward_euler(f, a, b, y0, h):
    n = (int)((b-a)/h + 1)
    t = np.linspace(a,b,n)
    y = np.zeros(n)
    y[0] = y0 # initial condition
    for i in range(n-1):
        y[i+1] = y[i] + h*f(t[i], y[i])

    return t, y

def implicit_euler(f, a, b, y0, h):
    n = (int)((b-a)/h + 1)
    t = np.linspace(a,b,n)
    y = np.zeros(n)
    y[0] = y0 # initial condition


# def newton():



#----------test-----------
def test():
    test_ode = ode(lambda t, y: -100*(y - np.sin(t)) + np.cos(t))
    test_ode.forward_euler(0, 1, 2, 0.01)
    test_ode.plot()

def test2():
    # step size h = 0.01
    t, y2 = forward_euler(lambda t, y2: -100*(y2 - np.sin(t)) + np.cos(t), 0, 1, 2, 0.01)
    _, y1 = forward_euler(lambda t, y1: -y1, 0, 1, 1, 0.01)

    t2, Y2 = forward_euler(lambda t2, Y2: -100*(Y2 - np.sin(t2)) + np.cos(t2), 0, 1, 2, 0.05)
    _, Y1 = forward_euler(lambda t2, Y1: -Y1, 0, 1, 1, 0.05)

    plt.figure(1)

    plt.subplot(321)
    plt.plot(t ,y2)
    plt.xlabel('t')
    plt.ylabel('y2(t)')
    plt.title('step size h = 0.01')

    plt.subplot(322)
    plt.plot(t2 ,Y2)
    plt.xlabel('t')
    plt.ylabel('y2(t)')
    plt.title('step size h = 0.05')


    plt.subplot(323)
    plt.plot(t, y1)
    plt.xlabel('t')
    plt.ylabel('y1(t)')

    plt.subplot(324)
    plt.plot(t2 ,Y1)
    plt.xlabel('t')
    plt.ylabel('y1(t)')

    plt.subplot(325)
    plt.plot(y1, y2)
    plt.xlabel('y1(t)')
    plt.ylabel('y2(t)')

    plt.subplot(326)
    plt.plot(Y1 ,Y2)
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



if __name__ == '__main__':
    test2()
