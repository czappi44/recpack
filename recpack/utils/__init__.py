import csv
from collections import defaultdict

import numpy as np
import scipy.sparse
import pandas as pd

from recpack.utils.globals import USER_KEY, ITEM_KEY, VALUE_KEY

import logging

logger = logging.getLogger("recpack")


def groupby2(keys, values):
    """ A group by of separate lists where order doesn't matter. """
    multidict = defaultdict(list)
    for k, v in zip(keys, values):
        multidict[k].append(v)
    return multidict.items()


def to_tuple(el):
    """ Whether single element or tuple, always returns as tuple. """
    if type(el) == tuple:
        return el
    else:
        return (el, )


def dict_to_csv(d, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(d.keys())
        writer.writerow(d.values())


def df_to_sparse(df, item_ix, user_ix, value_ix=None, shape=None):
    if value_ix is not None and value_ix in df:
        values = df[value_ix]
    else:
        if value_ix is not None:
            # value_ix provided, but not in df
            logger.warning(f"Value column {value_ix} not found in dataframe. Using ones instead.")

        num_entries = df.shape[0]
        # Scipy sums up the entries when an index-pair occurs more than once,
        # resulting in the actual counts being stored. Neat!
        values = np.ones(num_entries)

    indices = list(zip(*df.loc[:, [user_ix, item_ix]].values))

    if indices == []:
        indices = [[], []]  # Empty zip does not evaluate right

    if shape is None:
        shape = df[user_ix].max() + 1, df[item_ix].max() + 1
    sparse_matrix = scipy.sparse.csr_matrix(
        (values, indices), shape=shape, dtype=values.dtype
    )

    return sparse_matrix


class ItemCSVWriter(object):
    def __init__(self, item_id_mapping=None):
        super().__init__()
        self.item_id_mapping = item_id_mapping
        self.reverse_item_id_mapping = dict()
        if self.item_id_mapping is not None:
            pass

    def reverse_map_item_id(self, iid):
        return self.reverse_item_id_mapping.get(iid, iid)


class UserCSVWriter(object):
    def __init__(self, user_id_mapping=None):
        super().__init__()
        self.user_id_mapping = user_id_mapping
        self.reverse_user_id_mapping = dict()
        if self.user_id_mapping is not None:
            pass

    def reverse_map_user_id(self, uid):
        return self.reverse_user_id_mapping.get(uid, uid)


class InteractionsCSVWriter(UserCSVWriter, ItemCSVWriter):
    def __init__(self, user_id_mapping=None, item_id_mapping=None):
        UserCSVWriter.__init__(self, user_id_mapping=user_id_mapping)
        ItemCSVWriter.__init__(self, item_id_mapping=item_id_mapping)

    def reverse_map_user_item_id(self, uid, iid):
        return self.reverse_map_user_id(uid), self.reverse_map_item_id(iid)

    def sparse_to_csv(self, m, path, values=True):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            coo = m.tocoo()
            if values:
                writer.writerow([USER_KEY, ITEM_KEY, VALUE_KEY])
                for u, i, v in zip(coo.row, coo.col, coo.data):
                    uid, iid = self.reverse_map_user_item_id(u, i)
                    writer.writerow([uid, iid, v])
            else:
                writer.writerow([USER_KEY, ITEM_KEY])
                for u, i in zip(coo.row, coo.col):
                    uid, iid = self.reverse_map_user_item_id(u, i)
                    writer.writerow([uid, iid])


def csv_to_sparse(path, values=True):
    df = pd.read_csv(path)
    if values:
        return df_to_sparse(df, ITEM_KEY, USER_KEY, VALUE_KEY)
    else:
        return df_to_sparse(df, ITEM_KEY, USER_KEY)
