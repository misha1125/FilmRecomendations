import inspect
import time
import numpy as np
import pandas as pd
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ortho_group
from sklearn.datasets import make_sparse_spd_matrix as gen_spd


"""
n = 1024
eps = 1e-6

np.random.seed(0)
Q = ortho_group.rvs(n)
lambd = np.random.sample(n) * 20 + 1
eig_v = np.diag(lambd)
print(np.sqrt(np.max(lambd) / np.min(lambd)))

A = Q @ eig_v @ Q.T
M = np.eye(n)
M_inv = np.linalg.inv(M)

b = np.random.sample(n) * 1e5 - 1e5/2
x = np.zeros(n)
r = b - A @ x
r_m = M_inv @ r

d = np.copy(r)
for k in range(0, n + 1):
    r_product = r.T @ r_m
    alp = r_product / (d.T @ A @ d)
    x += alp * d
    r -= alp * (A @ d)
    r_m = M_inv @ r
    if np.linalg.norm(r_m) < eps:
        print(k)
        break
    beta = (r.T @ r_m) / r_product
    d = r_m + beta * d
print(np.linalg.norm(A @ x - b))

"""


def jacobi(A, b, x_init, epsilon=1e-10, max_iterations=10):
    D = np.diag(np.diag(A))
    LU = A - D
    x = x_init
    for i in range(max_iterations):
        D_inv = np.diag(1 / np.diag(D))
        x_new = np.dot(D_inv, b - np.dot(LU, x))
        if np.linalg.norm(x_new - x) < epsilon:
            return x_new
        x = x_new
    return x


def report(xk):
    global iter_number, data
    frame = inspect.currentframe().f_back
    data = np.append(data, [[iter_number], [frame.f_locals['resid']]], axis=1)
    iter_number += 1

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.callbacks = []
    def __call__(self, rk=None):
        self.callbacks.append(rk)
        self.niter += 1

counter_sgs = gmres_counter()
counter_icc = gmres_counter()
counter = gmres_counter()
counter_ilu = gmres_counter()

def icholesky(a):
    n = a.shape[0]
    for k in range(n):
        a[k, k] = np.sqrt(a[k, k])
        i_ = np.array(a[k + 1:, k].nonzero()).flatten()
        if len(i_) > 0:
            i_ += (k + 1)
            a[i_, k] /= a[k, k]
        for j in i_:
            i2_ = np.array(a[j:, j].nonzero()).flatten()
            if len(i2_) > 0:
                i2_ += j
                a[i2_, j] -= a[i2_, k] * a[j, k]
    return a

def test_cg(A, b, x0, M=None, tol=1e-8, label=None, draw=False):
    global iter_number, data, f, ax
    iter_number = 0
    data = np.empty(shape=(2, 0), dtype='float')
    start = time.time()
    if M is not None:
        spl.cg(A, b, x0=x0, M=M, callback=report, tol=tol)
    else:
        spl.cg(A, b, x0=x0, callback=report, tol=tol)
    end = time.time()
    if draw:
        draw(data, sct=False, label=label)
    return np.copy(data), end    - start

def draw(data, sct=False, label=None):
    global ax
    sns.lineplot(x=data[0], y=data[1], ax=ax, label=label)
    if sct:
        sns.scatterplot(x=data_original[0], y=data_original[1], ax=ax)

np.random.seed(1)
"""
n = 4
iter_o = np.empty(shape=(2, 0), dtype='float')
iter_sgs = np.empty(shape=(2, 0), dtype='float')
iter_j = np.empty(shape=(2, 0), dtype='float')
iter_icc = np.empty(shape=(2, 0), dtype='float')
iter_g_s = np.empty(shape=(2, 0), dtype='float')
iter_g_icc = np.empty(shape=(2, 0), dtype='float')
iter_g_ilu = np.empty(shape=(2, 0), dtype='float')
iter_g = np.empty(shape=(2, 0), dtype='float')
cond = np.empty(shape=0, dtype='float')
data = np.empty(shape=(2, 0), dtype='float')
data_original = None
data_SGS = None
data_ICC = None
"""
f, ax = plt.subplots(figsize=(7, 6))

n = 3
Q = ortho_group.rvs(n)
A = Q[:, :1]
x = np.random.sample((n, 1))
print(A / (A @ A.T @ x))
print()



while n <= 0:
    I = np.eye(n)
    iter_number = 0
    data = np.empty(shape=(2,0), dtype='float')

    """
    Q = ortho_group.rvs(n)
    spectr = np.random.sample(n) * 1e5 + 1
    spectr[0] = 1e5 + 1
    spectr[n - 1] = 1
    c_n = np.max(spectr) / np.min(spectr)
    D_s = np.diag(spectr)
    A = Q @ D_s @ Q.T
    """

    """
    spectr = np.random.sample(n) * 1e1 + 1
    spectr[0] = 1e1 + 1
    spectr[n - 1] = 1
    D_s = np.diag(spectr)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    A = U @ D_s @ V.T
    """

    #A = ortho_group.rvs(n)

    """
    Q = ortho_group.rvs(n)
    spectr = np.random.sample(n) * (1e5 / 2) - 1e5
    spectr[0] = 1e2
    spectr[1] = 1
    D_s = np.diag(spectr)
    A = Q @ D_s @ Q.T
    """

    """
    L = np.tril(np.random.sample((n, n)), k=-1)
    A = L - L.T
    """
    A = gen_spd(n, alpha=0.990, norm_diag=True, random_state=0)
    print("sparse coef:", np.count_nonzero(A) / n ** 2)
    c_n = np.linalg.cond(A)
    print("cond A:", c_n)
    cond = np.append(cond, c_n)
    b = np.random.sample(n)
    x0 = np.zeros(n)


    L_H = np.tril(icholesky(np.copy(A)))
    M_ICC = np.linalg.inv(L_H.T) @ np.linalg.inv(L_H)
    ILU = spl.spilu(A)
    M_ILU = spl.LinearOperator((n, n), ILU.solve)


    D = np.diag(np.diag(A))
    M_J = np.diag((1 / np.diag(A)))
    L = np.tril(A, k=-1)
    M_SGS = np.linalg.inv(L.T + D) @ D @ np.linalg.inv(L + D)

    #print("cond J", np.linalg.cond(M_J @ A))
    #print("cond SGS", np.linalg.cond(M_SGS @ A))
    #print("cond ICC", np.linalg.cond(M_ICC @ A))
    """ Full Jacobi
    iter_number = 0
    x_J = jacobi(A, b, x0)
    x = spl.cg(A, b, M=M_J, x0=x_J, callback=report, tol=1e-8)
    """
    print(n)

    data_original, time_o = test_cg(A, b, x0)
    iter_o = np.append(iter_o, [[n], [iter_number]], axis=1)
    print("in O:", iter_number, "time:",  time_o)
    data_SGS, time_sgs = test_cg(A, b, x0, M=M_SGS)
    iter_sgs = np.append(iter_sgs, [[n], [iter_number]], axis=1)
    print("in SGS:", iter_number, "time:",  time_sgs)

    """
    data_J, time_J = test_cg(A, b, x0, M=M_J)
    iter_j = np.append(iter_j, [[n], [iter_number]], axis=1)
    print("in J:", iter_number, "time:", time_J)
    """

    data_ICC, time_icc = test_cg(A, b, x0, M=M_ICC)
    print("in ICC:", iter_number, "time:",  time_icc)
    iter_icc = np.append(iter_icc, [[n], [iter_number]], axis=1)


    rs = 5000
    counter_sgs.niter = 0
    counter_icc.niter = 0
    counter.niter = 0
    counter_ilu.niter = 0


    start = time.time()
    x, info = spl.gmres(A, b, x0, restart=rs, M = M_SGS, callback=counter_sgs, tol=1e-8                                 )
    print("in GMRES_SGS:", counter_sgs.niter , "time:",  time.time() - start)
    print(np.linalg.norm(A @ x - b))
    iter_g_s = np.append(iter_g_s, [[n], [counter_sgs.niter]], axis=1)
    

    start = time.time()
    x, info = spl.gmres(A, b, x0, restart=rs, callback=counter, tol=1e-8)
    print("in GMRES:", counter.niter, "time:",  time.time() - start)
    iter_g = np.append(iter_g, [[n], [counter.niter]], axis=1)
    print(np.linalg.norm(A @ x - b))

    start = time.time()
    x, info = spl.gmres(A, b, x0, restart=rs, M = M_ICC, callback=counter_icc, tol=1e-8)
    print("in GMRES_ICC :", counter_icc.niter, "time:", time.time() - start)
    iter_g_icc = np.append(iter_g_icc, [[n], [counter_icc.niter]], axis=1)
    print(np.linalg.norm(A @ x - b))

    start = time.time()
    x, info = spl.gmres(A, b, x0, restart=rs, M=M_ILU, callback=counter_ilu, tol=1e-8)
    print("in GMRES_ILU :", counter_ilu.niter, "time:", time.time() - start)
    iter_g_ilu = np.append(iter_g_ilu, [[n], [counter_ilu.niter]], axis=1)
    print(np.linalg.norm(A @ x - b))


    """
    sns.lineplot(x=data_original[0], y=data_original[1], ax=ax, label='Original')
    sns.scatterplot(x=data_original[0], y=data_original[1], ax=ax)
    sns.lineplot(x=data_SGS[0], y=data_SGS[1], ax=ax, markers='True')
    sns.scatterplot(x=data_SGS[0], y=data_SGS[1], ax=ax)
    sns.lineplot(x=data_ICC[0], y=data_ICC[1], ax=ax, markers='True')
    sns.scatterplot(x=data_ICC[0], y=data_ICC[1], ax=ax)
    """
    """
 
    """

    n += 50


"""
#ax.set(xlabel='iterations', ylabel='residual')
ax.set(xlabel='n (size)', ylabel='k (iterations)')
#ax.set(yscale="log", xscale='log')

sns.lineplot(x=iter_o[0], y=iter_o[1], ax=ax, label='CG_original', markers=True)
sns.lineplot(x=iter_sgs[0], y=iter_sgs[1], ax=ax, markers='True', label='CG_SGS')
#sns.lineplot(x=iter_j[0], y=iter_j[1], ax=ax, markers='True', label='CG_J')
sns.lineplot(x=iter_icc[0], y=iter_icc[1], ax=ax, markers='True', label='CG_ICC')
cond = np.array([min(np.sqrt(cond[i]), np.max(iter_o[1]) + 10) for i in range(cond.shape[0])])
sns.lineplot(x=iter_o[0], y=cond, ax=ax, markers='True', label='sqrt(cond A)')
sns.lineplot(x=iter_g[0], y=iter_g[1], ax=ax, markers='True', label='GMERS')
sns.lineplot(x=iter_g_s[0], y=iter_g_s[1], ax=ax, markers='True', label='GMERS_SGS')
sns.lineplot(x=iter_g_icc[0], y=iter_g_icc[1], ax=ax, markers='True', label='GMERS_ICC')
sns.lineplot(x=iter_g_ilu[0], y=iter_g_ilu[1], ax=ax, markers='True', label='GMERS_ILU')
"""


"""
sns.lineplot(x=data_original[0], y=data_original[1], ax=ax, label='CG_original', markers=True)
sns.lineplot(x=data_SGS[0], y=data_SGS[1], ax=ax, markers='True', label='CG_SGS')
#sns.lineplot(x=data_ICC[0], y=data_ICC[1], ax=ax, markers='True', label='CG_ICC')
iter_gmr_o = np.array([i for i in range(len(counter.callbacks))])
sns.lineplot(x=iter_gmr_o, y=counter.callbacks, ax=ax, markers='True', label='GMRES')
iter_gmr_sgs = np.array([i for i in range(len(counter_sgs.callbacks))])
sns.lineplot(x=iter_gmr_sgs, y=counter_sgs.callbacks, ax=ax, markers='True', label='GMERS_SGS')
"""
"""
f.savefig('iterations12')
plt.show()
"""











