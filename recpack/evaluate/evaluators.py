from collections import defaultdict
from itertools import groupby

import numpy
import scipy.sparse


def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError("Matrix given must be of CSR format.")
    csr.data[csr.indptr[row]: csr.indptr[row + 1]] = value


class Evaluator:
    pass


class LeavePOutCrossValidationEvaluator(Evaluator):
    # TODO: Implement

    def __init__(self, p):
        self.p = p


class FoldIterator:
    """
    Iterator to get results from a fold instance.
    Fold instance can be any of the Evaluator classes defined.
    """

    def __init__(self, fold_instance, batch_size=1):
        self._fi = fold_instance
        self._index = 0
        self._max_index = fold_instance.sp_mat_in.shape[0]
        self.batch_size = batch_size
        assert self.batch_size > 0  # Avoid inf loops

    def __next__(self):
        # While loop to make it possible to skip any cases where user has no items in in or out set.
        while True:
            # Yield multi line fold.
            if self._index < self._max_index:
                start = self._index
                end = self._index + self.batch_size
                # make sure we don't go out of range
                if end >= self._max_index:
                    end = self._max_index

                fold_in = self._fi.sp_mat_in[start:end]
                fold_out = self._fi.sp_mat_out[start:end]

                # Filter out users with missing data in either in or out.

                # Get row sum for both in and out
                in_sum = fold_in.sum(1)
                out_sum = fold_out.sum(1)
                # Rows with sum == 0 are rows that should be removed in both in and out.
                rows_to_delete = []
                for i in range(len(in_sum)):
                    if in_sum[i, 0] == 0 or out_sum[i, 0] == 0:
                        rows_to_delete.append(i)
                # Set 0 values for rows to delete
                if len(rows_to_delete) > 0:
                    for r_to_del in rows_to_delete:
                        csr_row_set_nz_to_val(fold_in, r_to_del, value=0)
                        csr_row_set_nz_to_val(fold_out, r_to_del, value=0)
                    fold_in.eliminate_zeros()
                    fold_out.eliminate_zeros()
                    # Remove rows with 0 values
                    fold_in = fold_in[fold_in.getnnz(1) > 0]
                    fold_out = fold_out[fold_out.getnnz(1) > 0]
                # If no matrix is left over, continue to next batch without returning.
                if fold_in.nnz == 0:
                    self._index = end
                    continue

                self._index = end
                # Get a list of users we return as recommendations
                users = numpy.array([i for i in range(start, end)])
                users = list(set(users) - set(users[rows_to_delete]))

                return fold_in, fold_out, users
            raise StopIteration


class FoldInPercentageEvaluator(Evaluator):
    """
    Evaluator which generates in matrices by taking `fold_in` percentage of items a user has seen,
    the expected output are the remaining items for that user.

    :param fold_in: The percentage of each user's interactions to add in the input matrix.
    :type fold_in: `float`

    :param seed: The seed for random operations in this class. Very useful when desiring reproducible results.
    :type seed: `int`

    :param batch_size: the number of rows to generate when iterating over the evaluator object.
                       Larger batch_sizes make it possible to optimise computation time.
    :type data: `int`

    """

    def __init__(self, fold_in, seed=None, batch_size=1):
        self.fold_in = fold_in

        self.sp_mat_in = None
        self.sp_mat_out = None

        self.batch_size = batch_size
        self.seed = seed if seed else 12345

    def split(self, tr_data, val_data, te_data, shape=None):
        """
        Split the input data based on the parameters set at construction.
        Returns 2 sparse matrices in and out
        """
        # Set shape based on input data size
        if not shape:
            shape = tr_data.shape
        assert shape == tr_data.shape
        assert shape == val_data.shape
        assert shape == te_data.shape

        # easiest if we extract sparse value matrix from test, since that is the only part of the data we will use.
        sp_mat = te_data.values

        U, I, V = scipy.sparse.find(sp_mat)

        items_by_user_dct = defaultdict(list)

        # Unsure if U and I are sorted, which is a requirement for itertools.groupby.
        # So add them to a defaultdict to be sure
        groups = groupby(zip(U, I, V), lambda x: x[0])
        for key, subiter in groups:
            items_by_user_dct[key].extend(subiter)

        in_fold, out_fold = [], []

        # Seed random
        numpy.random.seed(self.seed)

        for u in items_by_user_dct.keys():
            usr_hist = items_by_user_dct[u]

            _len = len(usr_hist)

            numpy.random.shuffle(usr_hist)

            cut = int(numpy.floor(_len * self.fold_in))

            in_fold.extend(usr_hist[:cut])
            out_fold.extend(usr_hist[cut:])

        U_in, I_in, V_in = zip(*in_fold)

        self.sp_mat_in = scipy.sparse.csr_matrix((V_in, (U_in, I_in)), shape=shape)

        U_out, I_out, V_out = zip(*out_fold)
        self.sp_mat_out = scipy.sparse.csr_matrix((V_out, (U_out, I_out)), shape=shape)

        return self.sp_mat_in, self.sp_mat_out

    def __iter__(self):
        return FoldIterator(self, self.batch_size)


class TrainingInTestOutEvaluator(Evaluator):
    """
    Class to evaluate an algorithm by using training data as input, and test data as expected output.

    :param batch_size: the number of rows to generate when iterating over the evaluator object.
                       Larger batch_sizes make it possible to optimise computation time.
    :type data: `int`
    """

    def __init__(self, batch_size=1):

        self.sp_mat_in = None
        self.sp_mat_out = None

        self.batch_size = batch_size

    def split(self, tr_data, val_data, te_data, shape=None):
        """
        Split the data into in and out matrices.
        The in matrix will be the training data, and the out matrix will be the test data.
        """
        if not shape:
            shape = tr_data.shape
        assert shape == tr_data.shape
        assert shape == val_data.shape
        assert shape == te_data.shape

        U_in, I_in, V_in = [], [], []
        U_out, I_out, V_out = [], [], []

        # We are mostly intersted in user indices in training data, these will be the rows
        # we need to return.
        nonzero_users = list(set(tr_data.values.nonzero()[0]))
        tr_sp_mat = tr_data.values
        te_sp_mat = te_data.values
        for user in nonzero_users:
            # Add the in data
            tr_u_nzi = tr_sp_mat[user].indices
            tr_u = [user] * len(tr_u_nzi)
            tr_vals = tr_sp_mat[user, tr_u_nzi].todense().A[0]

            # Add the out data
            te_u_nzi = te_sp_mat[user].indices
            te_u = [user] * len(te_u_nzi)
            te_vals = te_sp_mat[user, te_u_nzi].todense().A[0]

            if len(tr_u_nzi) > 0 and len(te_u_nzi) > 0:
                # Only use the user if they actually have data for both
                # In and out
                I_in.extend(tr_u_nzi)
                U_in.extend(tr_u)
                V_in.extend(tr_vals)

                I_out.extend(te_u_nzi)
                U_out.extend(te_u)
                V_out.extend(te_vals)

        self.sp_mat_in = scipy.sparse.csr_matrix((V_in, (U_in, I_in)), shape=shape)

        self.sp_mat_out = scipy.sparse.csr_matrix((V_out, (U_out, I_out)), shape=shape)

        return self.sp_mat_in, self.sp_mat_out

    def __iter__(self):
        return FoldIterator(self, self.batch_size)


class TimedSplitEvaluator(Evaluator):
    def __init__(self, t, batch_size=1):
        self.t = t
        self.batch_size = 1

    def __iter__(self):
        return FoldIterator(self, self.batch_size)

    def split(self, tr_data, val_data, te_data, shape=None):
        """
        Split the data into in and out matrices.
        The in matrix will be the training data, and the out matrix will be the test data.
        """
        if not shape:
            shape = tr_data.shape
        assert shape == tr_data.shape
        assert shape == val_data.shape
        assert shape == te_data.shape

        # Split test data into before t and after t
        # in will be before t
        # out will be after t.

        data_fold_in = te_data.timestamps_lt(self.t)
        data_fold_out = te_data.timestamps_gte(self.t)

        self.sp_mat_in = data_fold_in.values

        self.sp_mat_out = data_fold_out.values

        return self.sp_mat_in, self.sp_mat_out