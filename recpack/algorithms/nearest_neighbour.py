import warnings

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csr import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from recpack.data.matrix import Matrix, to_csr_matrix
from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.util import get_top_K_values
from recpack.algorithms.util import invert


class ItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """Item K Nearest Neighbours model.

    First described in 'Item-based top-n recommendation algorithms.'
    Deshpande, Mukund, and George Karypis,
    ACM Transactions on Information Systems (TOIS) 22.1 (2004): 143-177

    For each item the K most similar items are computed during fit.
    Similarity parameter decides how to compute the similarity between two items.
    Supported options are: ``"cosine"`` and ``"conditional_probability"``

    - Cosine similarity between item i and j is computed as
      the ``count(i and j) / (count(i)*count(j))``.
    - Conditional probablity of item i with j is computed
      as ``count(i and j) / (count(i))``.
      Note that this is a non-symmetric similarity measure.

    If normalize is True, the scores are normalized per center item,
    making sure the sum of each row in the similarity matrix is 1.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import ItemKNN

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        # We'll only keep the closest neighbour for each item.
        # Default uses cosine similarity
        algo = ItemKNN(K=1)
        # Fit algorithm
        algo.fit(X)

        # We can inspect the fitted model
        print(algo.similarity_matrix_.nnz)
        # 3

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    **Example with Conditional Probability**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import ItemKNN

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        # We'll only keep the closest neighbour for each item.
        # we set the similarity measure to conditional probability
        # And enable normalization
        algo = ItemKNN(K=2, similarity='conditional_probability', normalize=True)
        # Fit algorithm
        algo.fit(X)

        # We can inspect the fitted model
        print(algo.similarity_matrix_.nnz)
        # 6

        # Similarities were normalized, so each row in the similarity matrix
        # sums to 1
        print(algo.similarity_matrix_.sum(axis=1))
        # [[1], [1], [1]]

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param similarity: Which similarity measure to use,
        can be one of ["cosine", "conditional_probability"], defaults to "cosine"
    :type similarity: str, optional
    :param pop_discount: [description], defaults to False
    :type pop_discount: bool, optional
    :param normalize_X: Normalize rows in the interaction matrix so that the contribution of
        users who have viewed more items is smaller, defaults to False
    :type normalize_X: bool, optional
    :param normalize_sim: Normalize scores per row in the similarity matrix to counteract
        artificially large similarity scores when the predictive item is rare, defaults to False.
    :type normalize_sim: bool, optional
    :param normalize: DEPRECATED. Use normalize_sim instead.
        Defaults to False
    :type normalize: bool, optional
    :raises ValueError: If an unsupported similarity measure is passed.
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability"]
    """The supported similarity options"""

    def __init__(self, K=200, similarity: str = "cosine", pop_discount=False, normalize_X=False, normalize_sim=False, normalize=False):
        """
        [summary]

        :param K: [description], defaults to 200
        :type K: int, optional
        :param similarity: [description], defaults to "cosine"
        :type similarity: str, optional

        """
        super().__init__(K)

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"similarity {similarity} not supported")
        self.similarity = similarity

        if self.similarity != "conditional_probability" and pop_discount:
            warnings.warn("Argument pop_discount is incompatible with all similarity \
                functions except conditional probability. This argument will be ignored, \
                not popularity discounting will be applied.", DeprecationWarning)

        self.pop_discount = pop_discount

        if normalize:
            warnings.warn(
                "Use of argument normalize is deprecated. Use normalize_sim instead.")

        self.normalize_X = normalize_X
        # Sim_normalize takes precedence.
        self.normalize_sim = normalize_sim if normalize_sim else normalize

        self.normalize = normalize

    def _compute_conditional_probability(self, X: csr_matrix) -> csr_matrix:
        # Cooccurence matrix
        co_mat = X.T @ X

        # Adding 1 additive smoothing to occurrences to avoid division by 0
        A = invert(diags(co_mat.diagonal()).tocsr())

        # We're trying to get a matrix S of P(j|i) where j is the column index,
        # i is the row index, so that we can later do X * S to obtain predictions.

        if self.pop_discount:
            # This has all item similarities
            item_cond_prob_similarities = A @ co_mat @ A.power(
                self.pop_discount)
        else:
            item_cond_prob_similarities = A @ co_mat
        # Set diagonal to 0, because we don't support self similarity
        item_cond_prob_similarities.setdiag(0)

        return item_cond_prob_similarities

    def _compute_cosine(self, X: csr_matrix) -> csr_matrix:
        # X.T otherwise we are doing a user KNN
        item_cosine_similarities = cosine_similarity(X.T, dense_output=False)
        item_cosine_similarities.setdiag(0)
        # Set diagonal to 0, because we don't want to support self similarity

        return item_cosine_similarities

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item"""
        transformer = Normalizer(norm="l1", copy=False)

        if self.normalize_X:
            X = transformer.transform(X)

        if self.similarity == "cosine":
            item_similarities = self._compute_cosine(X)
        elif self.similarity == "conditional_probability":
            item_similarities = self._compute_conditional_probability(X)

        item_similarities = get_top_K_values(item_similarities, self.K)

        # j, M (*, j) = 1
        if self.normalize_sim:
            # Normalize such that sum per row = 1
            item_similarities = transformer.transform(item_similarities)

        self.similarity_matrix_ = item_similarities


class ItemPNN(ItemKNN):
    """
    Item-based nearest neighbour method with probabilistic
    neighbour selection, rather than top-K.

    :param K: [description], defaults to 200
    :type K: int, optional
    :param similarity: [description], defaults to "cosine"
    :type similarity: str, optional
    :param pop_discount: [description], defaults to False
    :type pop_discount: bool, optional
    :param normalize_X: [description], defaults to False
    :type normalize_X: bool, optional
    :param normalize_sim: [description], defaults to False
    :type normalize_sim: bool, optional
    """

    def __init__(self, K=200, similarity: str = "cosine", pop_discount=False, normalize_X=False, normalize_sim=False):

        super().__init__(K=K, similarity=similarity, pop_discount=pop_discount,
                         normalize_X=normalize_X, normalize_sim=normalize_sim, )

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item"""
        transformer = Normalizer(norm="l1", copy=False)

        if self.normalize_X:
            X = transformer.transform(X)

        if self.similarity == "cosine":
            item_similarities = self._compute_cosine(X)
        elif self.similarity == "conditional_probability":
            item_similarities = self._compute_conditional_probability(X)

        item_similarities = get_top_K_values(item_similarities, self.K)

        # j, M (*, j) = 1
        if self.normalize_sim:
            # Normalize such that sum per row = 1
            item_similarities = transformer.transform(item_similarities)

        self.similarity_matrix_ = item_similarities

    def _predict(self, X: csr_matrix) -> csr_matrix:
        pass


def get_K_values(X, K, pdf):
    pass
