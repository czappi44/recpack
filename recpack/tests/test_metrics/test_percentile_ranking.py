import numpy
from scipy.sparse import csr_matrix

from recpack.metrics import PercentileRanking


def test_perc_ranking():
    values = [5, 1, 2, 4]
    users = [0, 0, 0, 0]
    items = [0, 2, 3, 7]
    y_true = csr_matrix((values, (users, items)), shape=(1, 10))

    values_pred = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    users_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    items_pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_pred = csr_matrix((values_pred, (users_pred, items_pred)), shape=(1, 10))

    pr = PercentileRanking()
    pr.calculate(y_true, y_pred)

    manual_numerator = 5 * 0.0 + 1 * 0.2 + 2 * 0.3 + 4 * 0.7
    manual_denominator = 5 + 1 + 2 + 4

    numpy.testing.assert_almost_equal(pr.value, manual_numerator / manual_denominator)