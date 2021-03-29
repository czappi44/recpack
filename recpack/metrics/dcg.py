import logging
import itertools

from scipy.sparse import csr_matrix
import numpy as np

from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero
from recpack.util import get_top_K_ranks

logger = logging.getLogger("recpack")


class DiscountedCumulativeGainK(ListwiseMetricK):
    """Discounted Cumulative Gain metric. Sum of cumulative gains.

    Discounted Cumulative Gain is computed for every user as

    .. math::

        DiscountedCumulativeGain(u) = \\sum\\limits_{i \\in TopK(u)} \\frac{y^{true}_{u,i}}{\\log_2 (\\text{rank}(u,i) + 1)}

    A single value is computed by taking the average over all users.

    :param K: Only topK of recommendations is used for calculate.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        denominator = y_pred_top_K.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        self.scores_ = csr_matrix(dcg.sum(axis=1))

        return


def dcg_k(y_true, y_pred, k=50):
    r = DiscountedCumulativeGainK(K=k)
    r.calculate(y_true, y_pred)

    return r.value


class NormalizedDiscountedCumulativeGainK(ListwiseMetricK):
    """Normalized Discounted Cumulative Gain metric.

    NormalizedDiscountedCumulativeGain is similar to DiscountedCumulativeGain,
    but normalises by dividing with the optimal,
    possible DiscountedCumulativeGain for the recommendation.
    Thus accounting for users where less than K items are available,
    and so the max score is lower than for other users.

    Scores are always in the interval [0, 1]

    .. math::

        \\text{NormalizedDiscountedCumulativeGain}(u) = \\frac{\\text{DCG}(u)}{\\text{IDCG}(u)}

    where ideal DiscountedCumulativeGain is

    .. math::

        \\text{IDCG}(u) = \\sum\\limits_{j=1}^{\\text{min}(K, |y^{true}_u|)} \\frac{1}{\\log_2 (j + 1)}

    :param K: How many of the top recommendations to consider.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

        self.discount_template = 1.0 / np.log2(np.arange(2, K + 2))
        # Calculate IDCG values by creating a list of partial sums (the
        # functional way)
        self.IDCG_cache = np.array(
            [1] + list(itertools.accumulate(self.discount_template, lambda x, y: x + y))
        )

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        # Correct predictions only
        denominator = y_pred_top_K.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        per_user_dcg = dcg.sum(axis=1)

        hist_len = y_true.sum(axis=1).astype(np.int32)
        hist_len[hist_len > self.K] = self.K

        self.scores_ = sparse_divide_nonzero(
            csr_matrix(per_user_dcg),
            csr_matrix(self.IDCG_cache[hist_len]),
        )

        return


def ndcg_k(y_true, y_pred, k=50):
    """Wrapper function around ndcg class.

    :param y_true: True labels
    :type y_true: csr_matrix
    :param y_pred: Predicted scores
    :type y_pred: csr_matrix
    :param k: top k to use for prediction, defaults to 50.
    :type k: int, optional
    :return: ndcg value
    :rtype: float
    """
    r = NormalizedDiscountedCumulativeGainK(K=k)
    r.calculate(y_true, y_pred)

    return r.value
