import scipy.sparse
import numpy as np
from sklearn.base import BaseEstimator


class Algorithm(BaseEstimator):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def identifier(self):
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    def __str__(self):
        return self.name

    def fit(self, X):
        pass

    def predict(self, X: scipy.sparse.csr_matrix, user_ids: np.array = None):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

class TwoTrainInputAlgorithm(Algorithm):
    def fit(self, X, Y):
        pass


class Ranker:
    def fit(self, X):
        pass
    
    def rank(
        self,
        scores: scipy.sparse.csr_matrix,
        X: scipy.sparse.csr_matrix,
        user_ids: np.array = None
    ):
        """Given the user history and the set of scores generated by some other algorithm, 
        the ranker will aim to change the order to give a better top K.
        """
        pass
