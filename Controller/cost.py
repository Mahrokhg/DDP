import math
import numpy as np


class Cost:
    def __init__(self, k, r, final_cost=0):
        self.k = k
        self.r = r
        self.final_cost = final_cost

    def l(self, x, u, final=False):
        if final:
            cost = self.final_cost + 1 - math.exp(self.k * (math.cos(x[0]) - 1))
        else:
            cost = 1 - math.exp(self.k * (math.cos(x[0]) - 1)) + self.r / 2 * u[0] ** 2
        return cost

    def l_x(self, x, u, final=False):
        temp = self.k * math.sin(x[0]) * math.exp(self.k*(math.cos(x[0]) - 1))
        return np.array([temp, 0])

    def l_u(self, x, u, final=False):
        return np.array([self.r * u[0]])

    def l_uu(self, x, u, final=False):
        return np.array([[self.r]])

    def l_ux(self, x, u, final=False):
        return np.array([[0, 0]])

    def l_xx(self, x, u, final=False):
        temp = self.k * (math.cos(x[0]) + math.sin(x[0])**2) * math.exp(self.k*(math.cos(x[0]) - 1))
        return np.array([[temp, 0],[0, 0]])
