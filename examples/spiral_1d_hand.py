"""SPIRAL in 1-D, small enough to do by hand.
Domain : channel between two plates, 0 <= y <= 1, fully developed flow:  d2u/dy2 = -2,  u(0) = 0, u(1) = 0  (exact: u = y(1-y))
Mesh   : 3 cells, h = 1/3, nodes y0=0 (wall, known), y1=1/3, y2=2/3 (unknown), y3=1 (wall, known)
Network: input y -> 2 hidden tanh nodes -> u = z1*a1 + z2*a2 ; z found by Newton on the finite-difference equations."""
import numpy as np
h = 1/3
y = np.array([1/3, 2/3])                      # INPUTS: positions of the unknown nodes
w = np.array([3.0, 3.0]); b = np.array([-1.0, -2.0])   # hidden weights (random in general, fixed here for easy numbers)
s = np.outer(y, w) + b                         # s_jk = w_k*y_j + b_k
Phi = np.tanh(s)                               # a_k(y_j)
def residual(u):                               # FD equation at each unknown node, walls u0 = u3 = 0 put in here (the BC)
    u_all = np.r_[0.0, u, 0.0]
    return np.array([(u_all[j-1] - 2*u_all[j] + u_all[j+1]) / h**2 + 2 for j in (1, 2)])
z = np.zeros(2); u = Phi @ z; r = residual(u)
print("s =\n", s, "\nPhi = tanh(s) =\n", Phi.round(5))
print("start: z =", z, " u =", u, " residual r =", r)
S = np.array([[-2, 1], [1, -2]]) / h**2       # d r / d u
J = S @ Phi                                    # d r / d z  (chain rule)
dz = np.linalg.solve(J, -r); z = z + dz; u = Phi @ z
print("J =\n", J.round(4), "\nNewton step dz =", dz.round(5), "\nu =", u.round(6), " residual =", residual(u).round(12))
print("classic FD solve:", np.linalg.solve(S, -2*np.ones(2)).round(6), "  exact y(1-y):", (y*(1-y)).round(6))
