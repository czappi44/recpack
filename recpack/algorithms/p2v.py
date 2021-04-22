import logging
import time
from typing import Tuple
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.data.matrix import InteractionMatrix, Matrix, to_binary, to_csr_matrix
from recpack.algorithms.samplers import sample_positives_and_negatives

logger = logging.getLogger("recpack")


class Prod2Vec(TorchMLAlgorithm):
    def __init__(self, embedding_size: int, negative_samples: int,
                 window_size: int, stopping_criterion: str, K=10, batch_size=1000, learning_rate=0.01, max_epochs=10,
                 prints_every_epoch=1,
                 stop_early: bool = False, max_iter_no_change: int = 5, min_improvement: float = 0.01, seed=None,
                 save_best_to_file=False, replace=False, exact=False):
        '''
        Prod2Vec algorithm from the paper: "E-commerce in Your Inbox: Product Recommendations at Scale". (https://arxiv.org/abs/1606.07154)

        Extends the TorchMLAlgorithm class.

        :param embedding_size: size of the embedding vectors
        :type embedding_size: int
        :param negative_samples: number of negative samples per positive sample
        :rtype negative_samples: int
        :param window_size: number of elements to the left and right of the target
        :rtype: int
        :param batch_size: How many samples to use in each update step.
        Higher batch sizes make each epoch more efficient,
        but increases the amount of epochs needed to converge to the optimum,
        by reducing the amount of updates per epoch.
        :type batch_size: int
        :param max_epochs: The max number of epochs to train.
            If the stopping criterion uses early stopping, less epochs could be used.
        :type max_epochs: int
        :param learning_rate: How much to update the weights at each update.
        :type learning_rate: float
        :param stopping_criterion: Name of the stopping criterion to use for training.
            For available values,
            check :meth:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
        :type stopping_criterion: str
        :param stop_early: If True, early stopping is enabled,
            and after ``max_iter_no_change`` iterations where improvement of loss function
            is below ``min_improvement`` the optimisation is stopped,
            even if max_epochs is not reached.
            Defaults to False
        :type stop_early: bool, optional
        :param max_iter_no_change: If early stopping is enabled,
            stop after this amount of iterations without change.
            Defaults to 5
        :type max_iter_no_change: int, optional
        :param min_improvement: If early stopping is enabled, no change is detected,
            if the improvement is below this value.
            Defaults to 0.01
        :type min_improvement: float, optional
        :param seed: Seed to the randomizers, useful for reproducible results,
            defaults to None
        :type seed: int, optional
        :param save_best_to_file: If true, the best model will be saved after training,
            defaults to False
        :type save_best_to_file: bool, optional
        :param K: for top-K most similar items
        :rtype k: int
        :param prints_every_epoch: when to print epoch loss
        :rtype prints_every_epoch: int
        :param replace: sample with or without replacement (see sample_positives_and_negatives)
        :rtype replace: bool
        :param exact: If False (default) negatives are checked against the corresponding positive sample only, allowing for (rare) collisions.
        If collisions should be avoided at all costs, use exact = True, but suffer decreased performance. (see sample_positives_and_negatives)
        :rtype exact: bool
        '''

        super(Prod2Vec, self).__init__(batch_size, max_epochs, learning_rate, stopping_criterion, stop_early,
                                       max_iter_no_change, min_improvement, seed, save_best_to_file)
        self.embedding_size = embedding_size
        self.negative_samples = negative_samples
        self.window_size = window_size
        self.prints_every_epoch = prints_every_epoch
        self.similarity_matrix_ = None
        self.K = K
        self.replace = replace
        self.exact = exact

    def _init_model(self, X: Matrix) -> None:
        self.model_ = SkipGram(X.shape[1], self.embedding_size)
        self.optimizer = optim.Adam(
            self.model_.parameters(), lr=self.learning_rate)
        self.epoch = 0

    def _evaluate(self, val_in: csr_matrix, val_out: csr_matrix) -> None:
        predictions = self._batch_predict(val_in)
        better = self.stopping_criterion.update(val_out, predictions)
        if better:
            logger.info("Model improved. Storing better model.")
            self._save_best()

    def _train_epoch(self, X: csr_matrix) -> None:
        assert self.model_ is not None
        start = time.time()
        epoch_loss = 0.0
        n_batches = 0
        # generator will just be restarted for each epoch.
        generator = self._training_generator(
            X, self.negative_samples, self.window_size, batch=self.batch_size)
        for focus_batch, positives_batch, negatives_batch in generator:
            focus_vectors = self.model_.get_embeddings(focus_batch)
            positives_vectors = self.model_.get_embeddings(positives_batch)
            negatives_vectors = self.model_.get_embeddings(negatives_batch)
            loss = self.model_.negative_sampling_loss(
                focus_vectors, positives_vectors, negatives_vectors)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1
        # the epoch loss with averaging over number of batches
        epoch_loss = epoch_loss / n_batches
        end = time.time()
        self.epoch += 1
        # Note: would it be cleaner to have a printing option in TorchMLAlgorithm fit?
        if self.epoch % self.prints_every_epoch == 0:
            print(
                f'Epoch [{self.epoch}/{self.max_epochs}] (in {int((end - start))}s) Epoch Loss: {epoch_loss:.4f}')

    def _batch_predict(self, X: csr_matrix):
        # need K+1 since we set diagonal to zero after calculating top-K
        K = self.K + 1
        embedding = self.model_.embeddings.weight.detach().numpy()
        num_items = embedding.shape[0]
        if K > num_items:
            K = num_items
            warnings.warn("K is larger than the number of items.", UserWarning)
        item_cosine_similarity_ = scipy.sparse.lil_matrix(
            (num_items, num_items))
        for batch in range(0, num_items, self.batch_size):
            Y = embedding[batch:batch + self.batch_size]
            item_cosine_similarity_batch = csr_matrix(
                cosine_similarity(Y, embedding))

            indices = [(i, j) for i, best_items_row in
                       enumerate(np.argpartition(item_cosine_similarity_batch.toarray(), -K)) for j in
                       best_items_row[-K:]]

            mask = scipy.sparse.csr_matrix(([1 for i in range(len(indices))], (list(zip(*indices)))),
                                           shape=(Y.shape[0], num_items))

            item_cosine_similarity_batch = item_cosine_similarity_batch.multiply(
                mask)
            item_cosine_similarity_[
                batch:batch + self.batch_size] = item_cosine_similarity_batch
        # no self similarity, set diagonal to zero
        item_cosine_similarity_.setdiag(0)
        self.similarity_matrix_ = csr_matrix(item_cosine_similarity_)
        scores = X @ self.similarity_matrix_
        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)
        return scores

    def _predict(self, X: csr_matrix) -> csr_matrix:
        # todo currently there's no user batching implemented
        results = self._batch_predict(X)
        logger.debug(f"shape of response ({results.shape})")
        return results.tocsr()

    def _sorted_item_history(self, X: InteractionMatrix) -> list:
        # TODO don't think there is a sorted_item_history in recpack
        # Note: this is not a generator as similar methods in InteractionMatrix
        """
        Group sessions (products) by user.

        Returns a list of lists.
        Each list represents a user's sequence of purchased products.
        Each product is designated by it's iid.

        :param: X: InteractionMatrix
        :returns: list of lists
        """
        df = X._df[['iid', 'uid']]
        grouped_df = df.groupby(['uid'])['iid'].apply(list)
        return grouped_df.values.tolist()

    def _window(self, sequences, window_size):
        '''
        Will apply a windowing operation to a sequence of item sequences.

        Note: pads the sequences to make sure edge cases are also accounted for.

        :param sequences: iterable of iterables
        :param window_size: the size of the window
        :returns: windowed sequences
        :rtype: numpy.ndarray
        '''
        padded_sequences = [[np.NAN] * window_size +
                            s + [np.NAN] * window_size for s in sequences]
        w = [w.tolist() for sequence in padded_sequences if len(sequence) >= window_size for w in
             sliding_window_view(sequence, 2 * window_size + 1)]
        return np.array(w)

    def _training_generator(self, X: InteractionMatrix, negative_samples: int, window_size: int, batch=1000):
        ''' Creates a training dataset using the skipgrams and negative sampling method.

        First, the sequences of items (iid) are grouped per user (uid).
        Next, a windowing operation is applied over each seperate item sequence.
        These windows are sliced and re-stacked in order to create skipgrams of two items (pos_training_set).
        Next, a unigram distribution of all the items in the dataset is created.
        This unigram distribution is used for creating the negative samples.

        :param X: InteractionMatrix
        :param negative_samples: number of negative samples per positive sample
        :param window_size: the window size
        :param batch: the batch size
        :yield: focus_batch, positive_samples_batch, negative_samples_batch
        :rtype: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
        '''
        # group and window
        sequences = self._sorted_item_history(X)
        windowed_sequences = self._window(sequences, window_size)
        context = np.hstack(
            (windowed_sequences[:, :window_size], windowed_sequences[:, window_size + 1:]))
        focus = windowed_sequences[:, window_size]
        # stack
        positives = np.empty((0, 2), dtype=int)
        for col in range(context.shape[1]):
            stacked = np.column_stack((focus, context[:, col]))
            positives = np.vstack([positives, stacked])
        # remove any NaN valued rows (consequence of windowing)
        positives = positives[~np.isnan(positives).any(axis=1)].astype(int)
        # create csr matrix from numpy array
        focus_items = positives[:, 0]
        pos_items = positives[:, 1]
        values = [1] * len(focus_items)
        positives_csr = scipy.sparse.csr_matrix(
            (values, (focus_items, pos_items)), shape=(X.shape[1], X.shape[1]))
        co_occurrence = to_binary(positives_csr + positives_csr.T)
        co_occurrence = scipy.sparse.lil_matrix(co_occurrence)
        co_occurrence.setdiag(1)
        co_occurrence = co_occurrence.tocsr()
        yield from sample_positives_and_negatives(X=co_occurrence, U=negative_samples, batch_size=batch, replace=self.replace,
                                                  exact=self.exact, positives=positives)

    def _transform_fit_input(self, X: Matrix, validation_data: Tuple[Matrix, Matrix]):
        return X, to_csr_matrix(validation_data, binary=True)


class SkipGram(nn.Module):
    def __init__(self, vocab_size, vector_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, vector_size)

    def get_embeddings(self, inputs):
        vectors = self.embeddings(inputs)
        return vectors

    def negative_sampling_loss(self, focus_vectors, positive_vectors, negative_vectors):
        embedding_dim = self.embeddings.embedding_dim
        focus_vectors = focus_vectors.reshape(-1, embedding_dim, 1)
        positive_vectors = positive_vectors.reshape(-1, 1, embedding_dim)
        # Focus vector and positive vector calculation:
        loss = torch.bmm(positive_vectors, focus_vectors).sigmoid().log()
        # Focus vector and negative vectors calculation:
        neg_loss = torch.bmm(negative_vectors.neg(),
                             focus_vectors).sigmoid().log()
        neg_loss = neg_loss.squeeze().sum(1)
        return -(loss + neg_loss).mean()
