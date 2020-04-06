from .algorithm_base import TwoMatrixFitAlgorithm
import scipy.sparse
import numpy
import pandass
from snapy import MinHash, LSH


def per_item_inverse_sparse_mat(sp_mat: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Invert each non zero entry in the matrix.
    """
    # By only inverting data, we avoid division by 0 issues. 0 entries stay 0
    inv = sp_mat.copy()
    inv.data == 1/sp_mat.data

    return inv


def compute_lift(cooc: scipy.sparse.csr_matrix, occ: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Compute lift given a cooccurence and occurence matrix
    Cooc is an |I| x |I| matrix, occ is a |U| x |I| matrix
    returns an |I| x |I| matrix
    """
    # Make sure shapes are correct, otherwise computation will not work
    assert cooc.shape[0] == cooc.shape[1]
    assert cooc.shape[0] == occ.shape[1]

    # Compute the lift
    # P(A AND B) / (P(A) * P(B))
    # => (C(A AND B) * NR_USERS) / (C(A) * C(B))
    lift = cooc.multiply(cooc.shape[0])\
               .multiply(1/occ.sum(axis=0))\
               .multiply(1/occ.sum(axis=0).T)
    return lift


def pv_lift_over_pur_lif(pv_csr, pur_csr, min_sup=5):
    """
    Compute per item pair with enough view support
    the lift of pageviews divided by the lift of the consumptions.
    """
    # Compute cooc counts
    pur_cooc = pur_csr.T @ pur_csr
    pv_cooc = pv_csr.T @ pv_csr

    # Remove pv pairs below threshold
    pv_cooc[min_sup >= pv_cooc] = 0
    pv_cooc.eliminate_zeros()

    binary_pv_cooc = pv_cooc.copy()
    binary_pv_cooc.data[binary_pv_cooc.data > 0] = 1
    binary_pur_cooc = pur_cooc.copy()
    binary_pur_cooc.data[binary_pur_cooc.data > 0] = 1

    # Each pair that has been seen enough, is assumed to be purchased once.
    pur_cooc_lil = pur_cooc.tolil()
    pur_cooc_lil = pur_cooc_lil + binary_pv_cooc - binary_pur_cooc
    pur_cooc = pur_cooc_lil.tocsr()

    # Compute lift
    pur_lift = compute_lift(pur_cooc, pur_csr)
    pv_lift = compute_lift(pv_cooc, pv_csr)

    # invert purchase lift
    inv_pur_lift = per_item_inverse_sparse_mat(pur_lift)

    # compute mux
    # pv_lift / pur_lift
    return pv_lift.multiply(inv_pur_lift)


class MuxPredictor(TwoMatrixFitAlgorithm):
    """
    Predict if an item is mutex with the history of a user.
    mutex defined as: Based on the user history, they will not purchase this item in the future.
    Typically because of another item that makes it so this other item will not be purchased.

    This base model fits a model based on view and consumption interaction matrices
    TODO: This is currently unusable in the pipeline. If we want to use this one,
    we should make it possible to use the base class in a pipeline.
    """
    def __init__(self, percentile=90):
        self.perc = percentile

    def get_mux_model(self, mux_scores):
        self.mux_model = mux_scores.copy()
        self.mux_model.setdiag(0)

        # Get the top Nth percentile as mux
        cuttoff = numpy.percentile(self.mux_model.data, self.perc)

        self.mux_model.data[self.mux_model.data < cuttoff] = 0
        self.mux_model.eliminate_zeros()

    def fit(self, X_view: scipy.sparse.csr_matrix, X_consumption: scipy.sparse.csr_matrix):
        """
        Fit a model based on view and consumption_data
        """
        mux_scores = pv_lift_over_pur_lif(X_view, X_consumption)
        self.get_mux_model(mux_scores)

    def predict(self, X):
        if self.mux_model is None:
            raise RunTimeException("mux model has not been trained yet.")
        values = self.mux_model @ X.T
        values[values > 0] = 1  # Binarize to make it easier for downstream use
        return values.T


class LSHMutexPredictor:
    """
    Mutex predicting model based on metadata only.
    Model based on Local Sensitivity Hashing implemented in the SnaPy library.

    TODO: We have no support in the pipeline for metadata based models.
    TODO: The fit method takes a pandas dataframe as input, which could be the standard for metadata based model
          Not sure though.

    TODO: Currently only supports training on a single column of metadata,
          extending this to a set of columns could improve the model

    """
    def __init__(self, shape, min_jaccard=0.3, n_gram=3, content_key='title', label_key='itemId'):
        self.mux_model = None

        self.n_gram = n_gram
        self.min_jaccard = min_jaccard
        self.shape = shape
        self.content_key = content_key
        self.label_key = label_key

    def train(self, df: pandas.Dataframe):
        content = list(df[self.content_key])
        labels = list(df[self.label_key])

        minhash = MinHash(content, n_gram=3, n_gram_type='char', permutations=100, hash_bits=64, seed=42)
        lsh = LSH(minhash, labels, no_of_bands=50)

        # Construct a scipy sparse Mutex prediction model
        x_ids = []
        y_ids = []
        for label in labels:
            alternatives = lsh.query(label, min_jaccard=self.min_jaccard)
            x_ids.extend([label for i in range(len(alternatives))])
            y_ids.extend(alternatives)

        self.mux_model = scipy.sparse.csr_matrix((numpy.ones(len(x_ids)), (x_ids, y_ids)), shape=self.shape)

    def predict(self, X):
        if self.mux_model is None:
            raise RunTimeException("mux model has not been trained yet.")
        values = self.mux_model @ X.T
        values[values > 0] = 1  # Binarize to make it easier for downstream use
        return values.T


# FIXME: this can be reused for other filtering purposes, but it was easier to write this way :D
class MuxFilter:
    """
    Filter class, takes a mux predictor as build argument.
    Subtracts the mux predictors score (assumed binary), from 1. Which makes it so mutex items have score 0,
    and simple pointwise mutliplication with a set of recommendations will remove the mutex items.

    TODO: No fit method, might be useful to change that,
          and have this class call the fit method of the underlying predictor

    TODO: No support in pipeline for filtering / reranking, will be needed to apply this in a standard way
          Might be good to implement this Filter as a true reranker.
          Takes as input for predict, history + recommendation matrix
    """
    def __init__(self, mux_predictor):
        self.mux_predictor = mux_predictor

    def predict(self, X):
        """
        Invert the predict method from the mux predictor. If it predicts 1, this returns 0, and the other way around.
        """
        values = self.mux_predictor.predict(X)
        return (numpy.ones(values.shape) - values)