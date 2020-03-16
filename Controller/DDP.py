import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from cost import *
from dynamics import *


class DDP:
    def __init__(self, dynamics, cost, N, max_iter=20, tol=1e-3):
        """Constructs an iLQR solver.
        Args:
            dynamics: Plant dynamics
            cost: Cost function
            N: Horizon length
            max_iter: Maximum iterations of backward and forward passes
            tol: Convergence criterion
        """
        self.dynamics = dynamics
        self.cost = cost
        # Solver param
        self.N = 10
        self.max_iter = max_iter
        self.tol = tol
        self.converged = False
        self.diverged = False
        # Regularization Schedule(II.F)
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = 1e9
        self._delta_0 = 1.2
        self._delta = self._delta_0
        # Backtrack line search parameter
        self.alphas = np.linspace(1, 0, num=20, endpoint=False)
        # Controller gains
        self._k = np.zeros((N, dynamics.action_size))
        self._K = np.zeros((N, dynamics.action_size, dynamics.state_size))
        # Trajectory
        self.x_cur = np.empty((N + 1, dynamics.state_size))
        self.u_cur = np.zeros((1, dynamics.state_size))
        # Derivative matrices
        self.F_x = np.empty((N, dynamics.state_size, dynamics.state_size))
        self.F_u = np.empty((N, dynamics.state_size))
        self.F_xx = np.empty((N, dynamics.state_size, dynamics.state_size, dynamics.state_size))
        self.F_ux = np.empty((N, dynamics.state_size, dynamics.state_size))
        self.F_uu = np.empty((N, dynamics.state_size))
        # Cost matrices
        self.j_opt = float("inf")
        self.L = np.empty(N + 1)
        self.L_x = np.empty((N + 1, dynamics.state_size))
        self.L_u = np.empty((N, dynamics.action_size))
        self.L_xx = np.empty((N + 1, dynamics.state_size, dynamics.state_size))
        self.L_ux = np.empty((N, dynamics.action_size, dynamics.state_size))
        self.L_uu = np.empty((N, dynamics.action_size, dynamics.action_size))

    def control(self, x0, us):
        """
        1. Initialize trajectory
        2. Backward pass
        3. Back-track line search
        4. Forward pass
        Stops when converged or reaches iterMax
        """
        # 1. Initialize trajectory
        self._init_traj(x0, us)
        # Iterations
        for i in range(self.max_iter):
            print('Iteration ', i)
            # 2. Backward pass
            self._backward_pass()
            if self.diverged:
                print('[Error] Unsuccessful iteration.')
                break
            # 3. Back-track line search & 4. Forward pass
            self._line_search()
            # Check for convergence
            if self.converged:
                break
        return self.x_cur, self.u_cur

    def _line_search(self):
        """
        Backtracking line search.
        Only performs a forward pass if j_new < j_opt.
        """
        flag = False
        for alpha in self.alphas:
            x_new, u_new = self._forward_pass(alpha)
            j_new = self._trajectory_cost(x_new, u_new)
            if j_new < self.j_opt:
                # Check for convergence
                if np.abs((self.j_opt - j_new) / self.j_opt) < self.tol:
                    self.converged = True
                flag = True
                self.j_opt = j_new
                self.x_cur = x_new
                self.u_cur = u_new
                # Decrease regularization term
                self._decrease_mu()
                # Move forward
                self._forward_pass(alpha)
                break
        if not flag:
            print('[Warning] The optimal J value was not updated in backtrack search.')
            # TODO

    def _backward_pass(self):
        # Check for non-PD
        _flag_quu_is_pd = False
        # Repeat until Q_uu is PD
        while not _flag_quu_is_pd:
            _flag_quu_is_pd = True
            # Control gains
            k = np.empty_like(self._k)
            K = np.empty_like(self._K)
            # Values
            v_x = self.L_x[-1].copy()
            v_xx = self.L_xx[-1].copy()
            # Move backwards
            for i in range(self.N-1, -1, -1):
                # Derivatives
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(v_x, v_xx, i)
                # If Q_uu is not PD
                if not np.all(np.linalg.eigvals(Q_uu) > 0):
                    # Increase regularization term
                    self._increase_mu()
                    _flag_quu_is_pd = False
                    print('[INFO] Non-PD Q_uu occurred at step ', i, '. mu is increased to ', self._mu, '.')
                    if self.diverged:
                        return
                    # TODO
                # Eq (6)
                k[i] = -np.linalg.solve(Q_uu, Q_u)
                K[i] = -np.linalg.solve(Q_uu, Q_ux)
                # Eq (11b).
                v_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
                v_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])
                # Eq (11c)
                v_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
                v_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
                v_xx = 0.5 * (v_xx + v_xx.T)  # To maintain symmetry.
        # Store the gains
        print('[INFO] Gains stored')
        self._k = np.array(k)
        self._K = np.array(K)

    def _forward_pass(self, alpha=1):
        # Initialize new X, U
        x_new = [self.x_cur]
        x_new[0] = self.x_cur[0]
        u_new = []
        # Compute new X,U
        for i in range(self.N):
            # Eq (12)
            u_new.append(self.u_cur[i] + alpha*self._k[i] + np.dot(self._K[i], (x_new[i] - self.x_cur[i])))
            # Eq (8.c)
            x_new.append(self.dynamics.f(x_new[i], u_new[i]))
        # Update X, U
        return x_new, u_new

    def _init_traj(self, x0, us):
        """Compute the trajectory from the starting
        state x0 by applying the control path us.
        Updates the derivative matrices.
        """
        self.u_cur = us
        self.x_cur[0] = x0
        for i in range(self.N):
            self.x_cur[i + 1] = self.dynamics.f(self.x_cur[i], us[i])
            # Dynamical derivatives
            self.F_x[i] = self.dynamics.f_x(self.x_cur[i], us[i])
            self.F_u[i] = self.dynamics.f_u(self.x_cur[i], us[i])
            self.F_xx[i] = self.dynamics.f_xx(self.x_cur[i], us[i])
            self.F_ux[i] = self.dynamics.f_ux(self.x_cur[i], us[i])
            self.F_uu[i] = self.dynamics.f_uu(self.x_cur[i], us[i])
            # Cost derivatives
            self.L[i] = self.cost.l(self.x_cur[i], us[i])
            self.L_x[i] = self.cost.l_x(self.x_cur[i], us[i])
            self.L_u[i] = self.cost.l_u(self.x_cur[i], us[i])
            self.L_xx[i] = self.cost.l_xx(self.x_cur[i], us[i])
            self.L_ux[i] = self.cost.l_ux(self.x_cur[i], us[i])
            self.L_uu[i] = self.cost.l_uu(self.x_cur[i], us[i])
        # Final cost
        self.L[-1] = self.cost.l(self.x_cur[-1], None, final=True)
        self.L_x[-1] = self.cost.l_x(self.x_cur[-1], None, final=True)
        self.L_xx[-1] = self.cost.l_xx(self.x_cur[-1], None, final=True)
        # J
        self.j_opt = self.L.sum()

    def _Q(self, V_x, V_xx, i):
        """Computes second order Q matrices.
        Args:
            V_x:
            V_xx:
            i: step number
        Returns:
            Q matrices
        """
        # Eqs (5a), (5b) and (5c)
        Q_x = self.L_x[i] + self.F_x[i].T.dot(V_x)
        Q_u = self.L_u[i] + self.F_u[i].T.dot(V_x)
        Q_xx = self.L_xx[i] + V_x.dot(self.F_xx[i]) + self.F_x[i].T.dot(V_xx).dot(self.F_x[i])
        # Eqs (11b) and (11c)
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_uu = self.L_uu[i] + V_x.dot(self.F_uu[i]) + self.F_u[i].T.dot(V_xx + reg).dot(self.F_u[i])
        Q_ux = self.L_ux[i] + V_x.dot(self.F_ux[i]) + self.F_u[i].T.dot(V_xx + reg).dot(self.F_x[i])
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def _trajectory_cost(self, x, u):
        cost = 0
        for i in range(self.N):
            cost += self.cost.l(x[i], u[i])
        # Final cost
        cost += self.cost.l(x[-1], None, final=True)
        return cost

    def _increase_mu(self):
        if self._mu == self._mu_max:
            print('[Warning:] Mu reached the maximum amount. Increase not possible.')
            self.diverged = True
        self._delta = max(1.0, self._delta) * self._delta_0
        self._mu = max(self._mu_min, self._mu * self._delta)
        self._mu = min(self._mu, self._mu_max)

    def _decrease_mu(self):
        self._delta = min(1.0, self._delta) / self._delta_0
        self._mu *= self._delta
        if self._mu <= self._mu_min:
            self._mu = 0.0
