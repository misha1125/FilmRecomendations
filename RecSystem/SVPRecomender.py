from RecSystem.BaceRecomender import BaceRecomender

import numpy as np
from numpy.linalg import norm


def svp_solve(A, mask, delta=0.1, epsilon=1e-2, max_iterations=1000, k=10):
    """
     A : m x n array
            matrix to complete
     mask : m x n array
     k = supposed rank
     delta = len of iterative step
    """

    X = np.zeros_like(A)
    # error vector to convergence visualisation
    error = []
    for t in range(max_iterations):
        Y = X - delta * mask * (X - A)
        U, S, V = np.linalg.svd(Y, full_matrices=False)
        S[k:] = 0
        X = np.linalg.multi_dot([U, np.diag(S), V])
        er = norm(mask * (X - A))
        error.append(er)
        if er < epsilon: break
    return X, error


class SVPRecomender(BaceRecomender):

    __score_matrix_rank__ = 10
    __optimal_delta__ = 0.55

    # score_matrix: numpy matrix n*m,
    # n = count of films
    # m = count of users
    # None = unknown
    def __init__(self, score_matric):
        super().__init__(score_matric)
        self.__optimal_delta__= self.__chose_optimat_delta__()
        self.__score_matrix_rank__ = self.__choose_score_matrix_rank__()

    # user_score_vector: numpy vector n*1
    def fill_system(self, user_score_vector):
        fill_matrix = np.append(self._score_matrix_, user_score_vector)
        mask = 1 - np.isnan(fill_matrix).astype(int)
        fill_matrix = np.nan_to_num(fill_matrix)
        filled_matrix = svp_solve(fill_matrix,mask,
                                  self.__optimal_delta__,0.1,1000,self.__score_matrix_rank__)
        return fill_matrix[-1]

    def __choose_score_matrix_rank__(self):
        return 10

    def __chose_optimat_delta__(self):
        return 0.55