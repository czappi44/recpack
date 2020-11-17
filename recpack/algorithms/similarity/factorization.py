import scipy.sparse
import sklearn.decomposition
from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.similarity.base import Algorithm, SimilarityMatrixAlgorithm
from recpack.data.matrix import Matrix, to_csr_matrix


class FactorizationAlgorithm(Algorithm):
    """Just a simple wrapper, for readability?

    :param Algorithm: [description]
    :type Algorithm: [type]
    """

    def predict(self, X):
        check_is_fitted(self)
        assert X.shape == (self.W_.shape[0], self.H_.shape[1])
        users = list(set(X.nonzero()[0]))
        result = scipy.sparse.lil_matrix(X.shape)
        result[users] = self.W_[users] @ self.H_
        return result.tocsr()


class NMF(FactorizationAlgorithm):
    # TODO check params NMF to see which ones are useful.
    def __init__(self, num_components=100, random_state=42):
        """NMF factorization implemented using the sklearn library.
        :param num_components: The size of the latent dimension
        :type num_components: int

        :param random_state: The seed for the random state to allow for comparison,
                             defaults to 42
        :type random_state: int, optional
        """
        super().__init__()
        self.num_components = num_components
        self.random_state = random_state

    def fit(self, X: Matrix):
        X = to_csr_matrix(X, binary=True)

        # Using Sklearn NMF implementation. For info and parameters:
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        model = sklearn.decomposition.NMF(
            n_components=self.num_components,
            init="random",
            random_state=self.random_state,
        )

        # Factorization is W * H. Where W contains user latent vectors, and H
        # contains item latent vectors
        self.W_ = model.fit_transform(X)
        self.H_ = model.components_

        # Post conditions
        assert self.W_.shape == (X.shape[0], self.num_components)
        assert self.H_.shape == (self.num_components, X.shape[1])

        return self


class NMFItemToItem(SimilarityMatrixAlgorithm):
    def __init__(self, num_components=100, random_state=42):
        super().__init__()
        self.num_components = num_components
        self.random_state = random_state

    def fit(self, X):
        self.model_ = NMF(self.num_components, self.random_state)
        self.model_.fit(X)

        self.similarity_matrix_ = self.model_.H_.T @ self.model_.H_

        self._check_fit_complete()


class SVD(FactorizationAlgorithm):
    """Singular Value Decomposition as dimension reduction recommendation algorithm.

    :param num_components: The size of the latent dimension
    :type num_components: int

    :param random_state: The seed for the random state to allow for comparison
    :type random_state: int
    """

    def __init__(self, num_components=100, random_state=42):
        super().__init__()

        self.num_components = num_components
        self.random_state = random_state

    def fit(self, X: Matrix):
        X = to_csr_matrix(X, binary=True)

        # TODO use other parameter options?
        model = sklearn.decomposition.TruncatedSVD(
            n_components=self.num_components, n_iter=7, random_state=self.random_state
        )
        # Factorization computes U x Sigma x V
        # U are the user features,
        # Sigma x V are the item features.
        self.W_ = model.fit_transform(X)

        V = model.components_
        sigma = scipy.sparse.diags(model.singular_values_)
        self.H_ = sigma @ V

        # Post conditions
        assert self.W_.shape == (X.shape[0], self.num_components)
        assert self.H_.shape == (self.num_components, X.shape[1])

        return self


class SVDItemToItem(SimilarityMatrixAlgorithm):
    def __init__(self, num_components=100, random_state=42):
        super().__init__()
        self.num_components = num_components
        self.random_state = random_state

    def fit(self, X):
        self.model_ = SVD(self.num_components, self.random_state)
        self.model_.fit(X)

        self.similarity_matrix_ = self.model_.H_.T @ self.model_.H_

        self._check_fit_complete()
