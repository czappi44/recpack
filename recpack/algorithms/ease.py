import numpy as np
import scipy.sparse

from recpack.algorithms.base import ItemSimilarityMatrixAlgorithm
from recpack.data.matrix import Matrix, to_csr_matrix


class EASE(ItemSimilarityMatrixAlgorithm):
    """Implementation of the EASEr algorithm.

    Implementation of the Embarrassingly Shallow Autoencoder as presented in
    Steck, Harald. "Embarrassingly shallow autoencoders for sparse data."
    The World Wide Web Conference. 2019.

    EASEr computes a dense item similarity matrix.
    Training aims to optimise the weights such that it can reconstruct the input matrix.

    Thanks to the closed form solution this algorithm has a significant speed up
    compared to the SLIM algorithm on which it is based.
    Memory consumption scales worse than quadratically in the amount of items.
    So check the size of the input matrix before using this algorithm.

    :param l2: regularization parameter to avoid overfitting, defaults to 1e3
    :type l2: float, optional
    :param alpha: parameter to punish popular items.
        Each similarity score between items i and j is divided by count(j)**alpha.
        Defaults to 0
    :type alpha: int, optional
    :param density: Parameter to reduce density of the output matrix,
        significantly speeds up and reduces memory footprint of prediction with a
        little loss of accuracy.
        Does not impact memory consumption of training.
        Defaults to None
    :type density: [type], optional
    """

    def __init__(self, l2=1e3, alpha=0, density=None):
        """[summary]

        :param l2: [description], defaults to 1e3
        :type l2: [type], optional
        :param alpha: [description], defaults to 0
        :type alpha: int, optional
        :param density: [description], defaults to None
        :type density: [type], optional
        """
        super().__init__()
        self.l2 = l2
        self.alpha = alpha  # alpha exponent for filtering popularity bias
        self.density = density

    def _fit(self, X: Matrix):
        """Compute the closed form solution,
        optionally rescalled to counter popularity bias (see param alpha)."""
        # Dense linear model algorithm with closed-form solution
        # Embarrassingly shallow auto-encoder from Steck @ WWW 2019
        # https://arxiv.org/pdf/1905.03375.pdf
        # Dense version in Steck et al. @ WSDM 2020
        # http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf
        # Eq. 21: B = I − P · diagMat(1 ⊘ diag(P)
        # More info on the solution for rescaling targets in section 4.2 of
        # Collaborative Filtering via High-Dimensional Regression from Steck
        # https://arxiv.org/pdf/1904.13033.pdf
        # Eq. 14 B_scaled = B * diagM(w)
        X = to_csr_matrix(X, binary=True)

        # Compute P
        XTX = (X.T @ X).toarray()
        P = np.linalg.inv(XTX + self.l2 * np.identity((X.shape[1]), dtype=np.float32))

        # Compute B
        B = np.identity(X.shape[1]) - P @ np.diag(1.0 / np.diag(P))
        B[np.diag_indices(B.shape[0])] = 0.0

        if self.alpha != 0:
            w = 1 / np.diag(XTX) ** self.alpha
            B = B @ np.diag(w)

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

        if self.density:
            self._prune()

    def _prune(self):
        # Prune B (similarity matrix)
        # Steck et al. state that we can increase the sparsity in matrix B without significant impact on quality.

        K = min(
            int(self.density * np.product(self.similarity_matrix_.shape)),
            self.similarity_matrix_.nnz,
        )
        self.similarity_matrix_.data[
            np.argpartition(abs(self.similarity_matrix_.data), -K)[0:-K]
        ] = 0
        self.similarity_matrix_.eliminate_zeros()


class EASE_XY(EASE):
    """Variation of EASE, reconstructing a second matrix given during training.

    Instead of autoencoding, trying to reconstruct the training matrix,
    training will try to construct the second matrix y, given the model and matrix X.

    :param l2: regularization parameter to avoid overfitting, defaults to 1e3
    :type l2: float, optional
    :param alpha: parameter to punish popular items.
        Each similarity score between items i and j is divided by count(j)**alpha.
        Defaults to 0
    :type alpha: int, optional
    :param density: Parameter to reduce density of the output matrix,
        significantly speeds up and reduces memory footprint of prediction with a
        little loss of accuracy.
        Does not impact memory consumption of training.
        Defaults to None
    :type density: [type], optional

    """

    def fit(self, X: Matrix, y: Matrix) -> "EASE_XY":
        """Fit the model, so it can predict interactions in matrix y, given matrix X

        :param X: Training data
        :type X: Matrix
        :param y: Matrix to predict
        :type y: Matrix, optional
        :return: self
        :rtype: EASE_XY
        """

        X, y = to_csr_matrix((X, y), binary=True)

        XTX = X.T @ X
        G = XTX + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ y).todense()

        D = np.diag(np.diag(B_rr) / np.diag(P))
        B = B_rr - P @ D

        if self.alpha != 0:
            w = 1 / np.diag(XTX.toarray()) ** self.alpha
            B = B @ np.diag(w)

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

        if self.density:
            self._prune()

        self._check_fit_complete()

        return self
