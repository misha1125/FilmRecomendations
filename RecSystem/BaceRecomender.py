from abc import ABC, abstractmethod
import numpy as np


class BaceRecomender(ABC):

    _score_matrix_ = np.matrix(10,10)
    def __init__(self,score_matric):
        _score_matrix_ = score_matric
        super().__init__()

    @abstractmethod
    def fill_system(self,user_score_vector):
        pass
