import logging

import numpy as np
import scipy.sparse
from recpack.util import get_top_K_ranks
from scipy.sparse import csr_matrix

from recpack.metrics.base import ElementwiseMetricK


logger = logging.getLogger("recpack")


class AUCAMAN(ElementwiseMetricK):
    """
    Measure is described in paper of S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme. Bpr: Bayesian
    personalized ranking from implicit feedback. In UAI, pages 452–461, 2009
    It is an AUC measure where a missing preference is evaluated as a dislike; given by the following formula:
        AUC_AMAN = 1 / |U_t| \\sum_{u \\in U_t}{\\frac{|I| - rank(h_u)}{|I| - 1}}
        where H_u is the set containing all test preferences of that user
    """

    def __init__(self):
        super().__init__(None)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """
        Calculate the AUC AMAN score for the particular y_true and y_pred matrices.
        :param y_true: User-item matrix with the actual true rating values.
        :param y_pred: User-item matrix with all prediction rating scores.
        :return: None: The result will be saved in self.value.
        """
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        _, items = y_true.shape
        assert items > 1

        y_pred_top_K = get_top_K_ranks(y_pred)
        self.y_pred_top_K_ = y_pred_top_K
        scores = scipy.sparse.lil_matrix(y_pred.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1
        # Element-wise multiplication. Result is a matrix with the rank of the predictions and the
        scores = scores.multiply(y_pred_top_K)
        scores.data = (items - scores.data) / (items - 1)

        scores = scores.tocsr()
        self.scores_ = scores

        # Value is the mean of the scores. 1 / |U_t| * sum_(u in U_t) scores
        self.value_ = scores.data.mean()

        return
