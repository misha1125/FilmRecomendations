import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from scipy.stats import ortho_group


def svp_solve(M, mask, A=None, B=None, tao=None, delta=None, epsilon=1e-4, max_iterations=300, r=10):
    global df
    if delta is None:
        delta = max(0.75 / density / 4, 1)
    print("svp_delta", delta)
    X = np.zeros_like(M)  # инициализируем X = 0
    t = 0
    fn_pr = np.linalg.norm(mask * M, ord='fro')
    for t in range(max_iterations):  # ограничиваем возможно количество итераций,
        # так как в некоторых случаях оно может быть очень большим
        Y = X - delta * mask * (X - M)  # вычисляем матрицу Y
        U, S, V = np.linalg.svd(Y, full_matrices=False)  # сингулярное разложение
        # матрицы Y
        S[r:] = 0  # берем только первые k сингулярных значений
        X = np.linalg.multi_dot([U, np.diag(S), V])  # перемножаем U*S*V^T
        it = df.shape[0]
        df.loc[it] = [t, 'known_svp', np.linalg.norm(mask * (X - M), ord='fro') / fn_pr]
        df.loc[it + 1] = [t, 'all_svp', np.linalg.norm(X - M, ord='fro') / fn_pr]

        if np.linalg.norm(mask * (M - X), ord='fro') / fn_pr < epsilon:
            break
    return X, t, t


def svt_solve(M, mask, tao, A=None, B=None, r=None, delta=None, epsilon=1e-2, max_iterations=300):
    global df
    if delta is None:
        delta = 1.2 / density
    print("svt_delta: ", delta)
    k_0 = int(np.ceil(tao / (delta * np.linalg.norm(mask * M, ord=2))))
    print("k_0: ", k_0)
    Y = k_0 * delta * mask * M

    fn_pr = np.linalg.norm(mask * M, ord='fro')
    t = 0
    for t in range(max_iterations):
        U, E, V = np.linalg.svd(Y)
        E_tao = np.maximum((E - tao), 0)
        X = np.linalg.multi_dot([U, np.diag(E_tao), V])
        Y = Y + delta * mask * (M - X)

        k = df.shape[0]
        df.loc[k] = [t, 'known_svt', np.linalg.norm(mask * (X - M), ord='fro') / fn_pr]
        df.loc[k + 1] = [t, 'all_svt', np.linalg.norm(X - M, ord='fro') / fn_pr]

        if np.linalg.norm(mask * (M - X), ord='fro') / fn_pr < epsilon:
            break
        """
        if np.linalg.norm(mask * (X - M), ord='fro') < epsilon:
            break
        """
    return X, t, t


def svt_wsinfo(M, A, B, mask, tao, delta=None, epsilon=1e-4, max_iterations=1000, r=None):
    global df

    def oper(Z):
        return mask * (A @ Z @ B.T)

    def oper_adj(X):
        return A.T @ (mask * X) @ B

    def comp(Z):
        return A.T @ (mask * (A @ Z @ B.T)) @ B

    if delta is None:
        delta = 1.2 / density
    print("svt_delta: ", delta)

    M_0 = oper_adj(mask * M)
    k_0 = int(np.ceil(tao / (delta * np.linalg.norm(M_0, ord=2))))
    print("k_0: ", k_0)

    Y = k_0 * delta * M_0

    fn_pr = np.linalg.norm(mask * M, ord='fro')
    t = 0
    for t in range(max_iterations):
        U, E, V = np.linalg.svd(Y)
        E_tao = np.maximum((E - tao), 0)
        Z = np.linalg.multi_dot([U, np.diag(E_tao), V])
        Y = Y + delta * (M_0 - comp(Z))

        k = df.shape[0]
        df.loc[k] = [t, 'known_svt_side', np.linalg.norm(mask * (A @ Z @ B.T - M), ord='fro') / fn_pr]
        df.loc[k + 1] = [t, 'all_svt_side', np.linalg.norm((A @ Z @ B.T - M), ord='fro') / fn_pr]
        # df.loc[k + 2] = [t, 'rank(Z)', np.linalg.norm(Z, ord=2)]
        if np.linalg.norm(mask * (A @ Z @ B.T - M), ord='fro') / fn_pr < epsilon:
            break

    return A @ Z @ B.T, t, t


def svp_wsinfo(M, A, B, mask, r=10, delta=None, epsilon=1e-4, max_iterations=1000, tao=None):
    global df

    def oper(Z):
        return mask * (A @ Z @ B.T)

    def oper_adj(X):
        return A.T @ (mask * X) @ B

    def comp(Z):
        return A.T @ (mask * (A @ Z @ B.T)) @ B

    def step_svp(delta):
        Y = Z + (delta) * (M_0 - comp(Z))
        U, E, V = np.linalg.svd(Y)

        E = E[:r]
        U = U[:, :r]
        V = V[:r, :]
        return np.linalg.multi_dot([U, np.diag(E), V])

    if delta is None:
        max_delta = max(0.75 / density, 1)
        delta = max_delta
    print("svp_max_delta: ", delta)
    delta_step_add = 0.1
    delta_step_div = 2

    M_0 = oper_adj(mask * M)

    fn_pr = np.linalg.norm(mask * M, ord='fro')
    Z = np.zeros_like(M_0)
    t = 0
    rt = 0
    last_error = np.inf
    while t <= max_iterations:
        rt += 1
        # пытаемся увеличить шаг
        Z_add = step_svp(delta + delta_step_add)
        error_add = np.linalg.norm(mask * (A @ Z_add @ B.T - M), ord='fro') / fn_pr
        if error_add < last_error:
            delta += delta_step_add
            Z = Z_add
        else:
            # не получилось, пробуем текущий шаг
            Z_zero = step_svp(delta)
            error_zero = np.linalg.norm(mask * (A @ Z_zero @ B.T - M), ord='fro') / fn_pr
            if error_zero < last_error:
                Z = Z_zero
            else:
                # всё плохо, уменьшаем шаг
                delta = max(delta / delta_step_div, 1)
                continue

        it = df.shape[0]
        error = np.linalg.norm(mask * (A @ Z @ B.T - M), ord='fro') / fn_pr
        df.loc[it] = [t, 'known_svp_side', error]
        df.loc[it + 1] = [t, 'all_svp_side', np.linalg.norm((A @ Z @ B.T - M), ord='fro') / fn_pr]
        df.loc[it + 2] = [t, 'delta', delta]
        t += 1
        last_error = error

        if np.linalg.norm(mask * (A @ Z @ B.T - M), ord='fro') / fn_pr < epsilon:
            break
    return A @ Z @ B.T, t, rt, delta


def test_method(name, complition, M, mask, A=None, B=None, r=10, delta=None, tao=None, epsilon=1e-4, mi=1000):
    print("-----------------------------------------")
    print(name)

    start = time.time()
    A_f, t_f, rt_f, d_f = complition(M=M, A=A, B=B, mask=mask, epsilon=epsilon, tao=tao, r=r, max_iterations=mi)
    time_alg = time.time() - start

    print("time: ", time_alg)
    print("iterations: ", t_f, rt_f)
    print("time for it: ", time_alg / t_f)
    print("final_delta", d_f)

    print("rank: ", np.linalg.matrix_rank(A_f))
    print("known error :", np.linalg.norm(mask * (A_f - M), ord='fro'))
    print("all error", np.linalg.norm((A_f - M), ord='fro'))
    print("-----------------------------------------")
    return A_f


f, ax = plt.subplots(figsize=(8, 7))
ax.set(xlabel='iterations', ylabel='errors')
df = pd.DataFrame(columns=['n_iter', 'error_type', 'error'])

np.random.seed(12312)
n = 1024
r = 10
L = 2 * np.random.sample((n, r)) - 1
R = 2 * np.random.sample((r, n)) - 1
A = L @ R
m = n ** 2 // 100
i = np.random.randint(low=0, high=n, size=m)
j = np.random.randint(low=0, high=n, size=m)
mask = np.zeros(shape=(n, n))
mask[i, j] = 1
m_real = mask.nonzero()[0].size

density = m_real / n ** 2
print("density: ", np.round(density, 3), m_real)

"""
U_s = ortho_group.rvs(n)[:, :r]
V_s = ortho_group.rvs(n)[:r, :]
E = np.array([1 / (2 ** k) for k in range(r)])
A = U_s @ np.diag(E) @ V_s
"""

U, E, V = np.linalg.svd(A)
U_s = U[:, :r]
V_s = V[:r, :]

U = U[:, : 2 * r]
V = V[: 2 * r, :]
Q_1 = ortho_group.rvs(2 * r)
Q_2 = ortho_group.rvs(2 * r)
U_2 = U @ Q_1
V_2 = Q_2 @ V



#A_svt_side = test_method("svt_sinfo", svt_wsinfo, M=A, A=U_s, B=V_s.T, mask=mask, epsilon=1e-4, tao=1000, mi=100)

A_svp_side = test_method("svp_sinfo", svp_wsinfo, M=A, A=U_2, B=V_2.T, mask=mask, epsilon=1e-4, r=r, mi=500)
#A_svp = test_method("svp", svp_solve, M=A, mask=mask, epsilon=1e-4, r=r, mi=300)
"""

U_c, E_c, V_c = np.linalg.svd(A_svt_side)
U_c = U_c[:, :r]
V_c = V_c[:r, :]
# print(np.linalg.norm(U[:, :r] - U_c))
# print(np.linalg.norm(V[:r, :] - V_c))
"""

"""
print("SVT: ")
start = time.time()
A_svt, t_svt = svt_solve(A, mask, epsilon=1e-2, tao=10000)
print("time: ", time.time() - start)
print("ieration: ", t_svt)

print(np.linalg.matrix_rank(A_svt))
print(np.linalg.norm(mask * (A_svt - A), ord='fro'))
print(np.linalg.norm((A_svt - A), ord='fro'))
"""
# print(df.error[df.error_type == 'all_svp'].min())
# print(df[(df.error_type == 'all_svp') | (df.error_type == 'known_svp')].tail(10))

# ax.set_ylim(ymin=df.error.min(), ymax=df.error.max())
sns.lineplot(data=df, ax=ax, x='n_iter', y='error', hue='error_type')
ax.set(yscale='log')

plt.show()
"""
r_a = 51
r_b = 211
A = np.random.sample((n, r_a))
B = np.random.sample((n, r_b))
Z = np.random.sample((r_a, r_b))
X = np.random.sample((n, n))
print(np.trace((mask * (A @ Z @ B.T)).T @ X))
print(np.trace((B @ Z.T @ A.T) @ (X * mask)))
print(np.trace(Z.T @ (A.T @ (X * mask) @ B)))
print(np.linalg.norm((mask * X - X * mask)))
"""