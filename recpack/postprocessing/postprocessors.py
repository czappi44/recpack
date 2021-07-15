"""Postprocessors apply filtering onto Recpack's internal output representations."""

import logging
from typing import List
from recpack.postprocessing.filters import PostFilter
from scipy.sparse import csr_matrix
import numpy as np

logger = logging.getLogger("recpack")


class Postprocessor:
    """Class to postprocess a CSR Response Matrix.

    Postprocessing applies filters to the given input (algorithm output) data.

    In order to apply filters they should be added using :meth:`add_filter`.
    The filters and the two other steps get applied during :meth:`process`
    and :meth:`process_many`.
    """

    def __init__(self):
        self.filters = []

    def add_filter(self, _filter: PostFilter, index: int = None):
        """ Add a post-processing filter.
        """
        if index is None:
            self.filters.append(_filter)
        else:
            self.filters.insert(index, _filter)

    def process(self, X_pred: csr_matrix) -> csr_matrix:
        """
        :param X_pred: csr_matrix containing recommended user-item pairs.
        :type X_pred: csr_matrix
        :return: csr_matrix-object containing the filtered response.
        :rtype: csr_matrix
        """
        return self.process_many(X_pred)[0]

    def process_many(self, *X_preds: csr_matrix) -> List[csr_matrix]:
        """
        Process all CSR Matrices passed as arguments.

        :param X_preds: The CSR matrices to process
        :type X_preds: csr_matrix
        :return: A list of csr_matrix objects in the order
            the csr_matrices were passed in.
        :rtype: List[csr_matrix]
        """

        for index, X_pred in enumerate(X_preds):
            logger.debug(f"Processing X_pred {index}")
            logger.debug(f"\tusers with recommendations before preprocess: {sum(np.diff(X_pred.indptr) != 0)}")

        for filter in self.filters:
            logger.debug(f"Applying post processing filter: {filter}")
            X_preds = filter.apply_all(*X_preds)
            for index, X_pred in enumerate(X_preds):
                logger.debug(f"X_pred {index}")
                logger.debug(f"\tusers with recommendations after preprocess: {sum(np.diff(X_pred.indptr) != 0)}")

        return X_preds
