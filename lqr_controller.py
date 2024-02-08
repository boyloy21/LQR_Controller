import numpy as np
import math 
import scipy.linalg as la


class LQR_Controller():
    def __init__(self, Q, R,A, B, N):
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.state_minus = np.array([[0.0], [0.0], [0.0]])
        self.N = N
        self.P = [None]*(self.N+1)
    def state_space(self,U):
        state = (self.A @ self.state_minus) + (self.B @ U)
        self.state_minus = state

    def Calculate_lqr(self, errorX, errorY, errorYaw):
        error = np.array([[errorX], [errorY], [errorYaw]])
        Qf = self.Q
        self.P[self.N] = Qf
        for i in range(self.N, 0 ,-1):
            self.P[i-1] = self.Q + self.A.T @ self.P[i] @ self.A - (self.A.T @ self.P[i] @ self.B) @ np.linalg.pinv(self.R + (self.B.T @ self.P[i] @ self.B)) @ (self.B.T @ self.P[i] @ self.A)

        K = [None] * self.N
        U = [None] * self.N

        for i in range(self.N):
            K[i] = np.linalg.inv(self.B.T @ self.P[i+1] @ self.B + self.R) @ (self.B.T @ self.P[i+1] @ self.A)
            U[i] = -K[i] @ error

        u_start = U[self.N-1]
        return u_start

class LQR_ControllerV2():
    def __init__(self,A,B,Q,R,max_iter):
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.max_iter = max_iter
    def dare(self):
        """solve a discrete time_Algebraic Riccati equation (DARE) """
        P = self.Q
        P_next = self.Q
        eps = 0.01
        for i in range(self.max_iter):
            self.P_next = self.A.T @ P @ self.A - self.A.T @ P @ self.B @ \
                    la.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A + self.Q
            if (abs(P_next - P)).max() < eps:
                break
            P = self.P_next

        return P_next
    def dlqr(self):
        # first, try to solve the ricatti equation
        P_Next = self.dare()

        # compute the LQR gain
        K = la.inv(self.B.T @ P_Next @ self.B + self.R) @ (self.B.T @ P_Next @ self.A)

        eig_result = la.eig(self.A - self.B @ K)

        return K, P_Next, eig_result[0]



if __name__=="__main__":
    LQR_Controller()
    LQR_ControllerV2()




