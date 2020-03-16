# Dynamic Model
import numpy as np
import math


class Dynamics:
    def __init__(self,
                 dt=1e-4,
                 a=10,
                 b=1,
                 s=1,
                 v_max=10,
                 u_max=1,
                 n_1=100,
                 n_2=100,
                 n_u=1e10
                 ):
        """Constructs an Inverted Pendulum model
                Args:

                Note:
                    state:
                    action:
                    theta: 0 is pointing up and increasing counter-clockwise.
                """
        self.dt = dt
        # Physical prop
        self.a = a
        self.b = b
        self.s = s
        self.v_max = v_max
        self.u_max = u_max
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_u = n_u
        # model
        self.state_size = 2
        self.action_size = 1

    def f(self, x, u):
        for i in range(self.action_size):
            # round control to nearest grid point
            u[i] = round(u[i] * self.n_u / self.u_max) / self.n_u * self.u_max
            # limit control
            u[i] = min(u[i], self.u_max)
            u[i] = max(u[i], -self.u_max)
        # new state
        x0_new = x[0] + self.dt * x[1]
        x1_new = x[1] + self.dt * (self.a * math.sin(x[0]) - self.b * x[1] + u[0])
        # limit speed
        x1_new = min(x1_new, self.v_max)
        x1_new = max(x1_new, -self.v_max)
        # round state to nearest grid point
        #x0_new = round(x0_new / self.n_1) * self.n_1
        #x1_new = round(x1_new / self.n_2) * self.n_2
        return np.array([x0_new, x1_new])

    def f_x(self, x, u):
        x_11 = 0
        x_12 = 1
        x_21 = self.a * math.cos(x[0])
        x_22 = -self.b
        return np.array([[x_11, x_12], [x_21, x_22]])

    def f_u(self, x, u):
        return np.array([0, 1])

    def f_xx(self, x, u):
        mat1 = np.array([[0, 0], [-self.a * math.sin(x[0]), 0]])
        mat2 = np.array([[0, 0], [0, 0]])
        return np.array([mat1, mat2])

    def f_xu(self, x, u):
        return np.array([[0, 0], [0, 0]])

    def f_ux(self, x, u):
        return np.array([0, 0])

    def f_uu(self, x, u):
        return np.array([0, 0])
