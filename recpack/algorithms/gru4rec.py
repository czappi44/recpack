import logging
from math import ceil
import time
from typing import Tuple, List, Iterator, Optional

from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import (
    bpr_loss,
    bpr_max_loss,
    top1_loss,
    top1_max_loss,
)
from recpack.algorithms.samplers import (
    SequenceMiniBatchPositivesTargetsNegativesSampler,
    SequenceMiniBatchSampler,
)
from recpack.matrix import InteractionMatrix


logger = logging.getLogger("recpack")


class GRU4Rec(TorchMLAlgorithm):
    """Base class for recurrent neural networks for session-based recommendations.

    The algorithm, also known as GRU4Rec, was introduced in the 2016 and 2018 papers
    "Session-based Recommendations with Recurrent Neural Networks" and
    "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations"

    The algorithm makes recommendations by training a recurrent neural network to
    predict the next action of a user, and using the most likely next actions as
    recommendations. At the heart of it is a Gated Recurrent Unit (GRU), a recurrent
    network architecture that is able to form long-term memories.

    Predictions are made by processing a user's actions so far one by one,
    in chronological order::

                                          iid_3_predictions
                                                  |
                 0 --> [ GRU ] --> [ GRU ] --> [ GRU ]
                          |           |           |
                        iid_0       iid_1       iid_2

    here 'iid' are item ids, which can represent page views, purchases, or some other
    action. The GRU builds up a memory of the actions so far and predicts what the
    next action will be based on what other users with similar histories did next.
    While originally devised to make recommendations based on (often short) user
    sessions, the algorithm can be used with long user histories as well.

    For the mathematical details of GRU see "Empirical Evaluation of Gated Recurrent
    Neural Networks on Sequence Modeling" by Chung et al.

    :param num_layers: Number of hidden layers in the RNN. Defaults to 1
    :type num_layers: int, optional
    :param hidden_size: Number of neurons in the hidden layer(s). Defaults to 100
    :type hidden_size: int, optional
    :param num_components: Size of item embeddings. Defaults to 250
    :type num_components: int, optional
    :param dropout: Dropout applied to embeddings and hidden layer(s).
        Defauls to 0 (no dropout)
    :type dropout: float
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad".
        Defaults to "adagrad"
    :type optimization_algorithm: str, optional
    :param momentum: Momentum when using the sgd optimizer.
        Defaults to 0.0
    :type momentum: float, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping.
        Defaults to 1.0
    :type clipnorm: float, optional
    :param bptt: Number of backpropagation through time steps.
        Defaults to 1
    :type bptt: int, optional
    :param num_negatives: Number of negatives to sample for every positive.
        Defaults to 0
    :type num_negatives: int, optional
    :param batch_size: Number of examples in a mini-batch.
        Defaults to 512.
    :type batch_size: int, optional
    :param max_epochs: Max training runs through entire dataset.
        Defaults to 5
    :type max_epochs: int, optional
    :param learning_rate: Gradient descent initial learning rate
        Defaults to 0.03
    :type learning_rate: float, optional
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
    :type stopping_criterion: str, optional
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
        Defaults to 0.0
    :type min_improvement: float, optional
    :param seed: Seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If true, the best model will be saved after training,
        defaults to False
    :type save_best_to_file: bool, optional
    :param keep_last: Retain last model, rather than best
        (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param predict_topK: The topK recommendations to keep per row in the matrix.
        Use when the user x item output matrix would become too large for RAM.
        Defaults to None, which results in no filtering.
    :type predict_topK: int, optional
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        num_components: int = 250,
        dropout_p_embed: float = 0,
        dropout_p_hidden: float = 0,
        optimization_algorithm: str = "adagrad",
        momentum: float = 0.0,
        clipnorm: float = 1.0,
        bptt: int = 1,
        num_negatives: int = 0,
        batch_size: int = 512,
        max_epochs: int = 5,
        learning_rate: float = 0.03,
        stopping_criterion: str = "recall",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed: Optional[int] = None,
        save_best_to_file: bool = False,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
        final_activation: str = None,
        use_correct_weight_init: bool = False,
    ):
        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_components = num_components
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.optimization_algorithm = optimization_algorithm
        self.momentum = momentum
        self.clipnorm = clipnorm
        self.bptt = bptt
        self.num_negatives = num_negatives
        self.final_activation = final_activation
        self.use_correct_weight_init = use_correct_weight_init

    def _init_model(self, X: InteractionMatrix) -> None:
        # Invalid item ID. Used to mask inputs to the RNN
        self.num_items = X.shape[1]
        self.pad_token = self.num_items

        self.model_ = GRU4RecTorch(
            self.num_items,
            self.hidden_size,
            self.num_components,
            self.pad_token,
            num_layers=self.num_layers,
            dropout_p_embed=self.dropout_p_embed,
            dropout_p_hidden=self.dropout_p_hidden,
            final_activation=self.final_activation,
            use_correct_weight_init = self.use_correct_weight_init,
        ).to(self.device)

        if self.optimization_algorithm == "sgd":
            self.optimizer = optim.SGD(self.model_.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.optimization_algorithm == "adagrad":
            self.optimizer = optim.Adagrad(self.model_.parameters(), lr=self.learning_rate)

        self.predict_sampler = SequenceMiniBatchSampler(self.pad_token, batch_size=self.batch_size)

        self.fit_sampler = SequenceMiniBatchPositivesTargetsNegativesSampler(
            self.num_negatives, self.pad_token, batch_size=self.batch_size
        )

    def _transform_fit_input(
        self,
        X: InteractionMatrix,
        validation_data: Tuple[InteractionMatrix, InteractionMatrix],
    ):
        """Transform the input matrices of the training function to the expected types

        :param X: The interactions matrix
        :type X: Matrix
        :param validation_data: The tuple with validation_in and validation_out data
        :type validation_data: Tuple[Matrix, Matrix]
        :return: The transformed matrices
        :rtype: Tuple[csr_matrix, Tuple[csr_matrix, csr_matrix]]
        """
        self._assert_is_interaction_matrix(X, *validation_data)
        self._assert_has_timestamps(X, *validation_data)

        return X, validation_data

    def _transform_predict_input(self, X: InteractionMatrix) -> InteractionMatrix:
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _train_epoch(self, X: InteractionMatrix) -> List[float]:
        losses = []
        minibatch_losses = []
        events = []
        num_sessions = X._df[X.USER_IX].nunique()
        for (
            _,
            positives_batch,
            targets_batch,
            negatives_batch,
        ) in tqdm(self.fit_sampler.sample(X), total=num_sessions//self.batch_size+1, leave=True, position = 0):
            st = time.time()
            positives_batch = positives_batch.to(self.device)
            targets_batch = targets_batch.to(self.device)
            negatives_batch = negatives_batch.to(self.device)

            logger.debug(f"Takes {time.time() - st} seconds to convert to GPU")

            batch_loss = 0
            true_batch_size = positives_batch.shape[0]
            # Want to reuse this between chunks of the same batch of sequences
            hidden = self.model_.init_hidden(true_batch_size).to(self.device)

            # Generate vertical chunks of BPTT width
            for p, (input_chunk, target_chunk, neg_chunk) in enumerate(self._chunk(
                self.bptt, positives_batch, targets_batch, negatives_batch
            )):
                input_mask = target_chunk != self.pad_token
                # Remove rows with only pad tokens from chunk and from hidden.
                # We can do this because the array is sorted.
                true_rows = input_mask.any(axis=1)
                true_input_chunk = input_chunk[true_rows]
                true_target_chunk = target_chunk[true_rows]
                true_neg_chunk = neg_chunk[true_rows]
                true_hidden = hidden[:, true_rows, :]
                true_input_mask = input_mask[true_rows]

                if true_input_chunk.shape[0] != 0:

                    self.optimizer.zero_grad()
                    output, hidden[:, true_rows, :] = self.model_(true_input_chunk, true_hidden)
                    loss = self._compute_loss(output, true_target_chunk, true_neg_chunk, true_input_mask)
                    loss.backward()
                    with torch.no_grad():
                        batch_loss += loss.item()
                        events.append(len(true_input_chunk))
                        minibatch_losses.append(loss.item())
                    if self.clipnorm:
                        nn.utils.clip_grad_norm_(self.model_.parameters(), self.clipnorm)

                    self.optimizer.step()

                    hidden = hidden.detach()
            logger.debug(f"Takes {time.time() - st} seconds to process batch")
            losses.append(batch_loss)

        return losses, minibatch_losses, events

    def _compute_loss(
        self,
        output: torch.FloatTensor,
        targets_chunk: torch.LongTensor,
        negatives_chunk: torch.LongTensor,
        true_input_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _chunk(self, chunk_size: int, *tensors: torch.LongTensor) -> Iterator[Tuple[torch.LongTensor, ...]]:
        """Split tensors into chunks of self.bptt width.
        Chunks can be of width self.bptt - 1 if max_hist_len % self. bptt != 0.

        :param tensors: Tensors to be chunked.
        :type tensors: torch.LongTensor
        :yield: Tuple of chunked tensors, one chunk of each of the input tensors.
        :rtype: Iterator[Tuple[torch.LongTensor, ...]]
        """
        max_hist_len = tensors[0].shape[1]

        num_chunks = ceil(max_hist_len / chunk_size)
        split_tensors = [t.tensor_split(num_chunks, axis=1) for t in tensors]

        return zip(*split_tensors)

    def _predict(self, X: InteractionMatrix, m):
        '''We assume that self.bptt == 1 as in the original paper
        '''
        X_pred = lil_matrix((X.shape[0], self.num_items))
        self.model_.eval()
        with torch.no_grad():
            # Loop through users in batches
            sum_rec, sum_mrr, measures = np.zeros_like(m, dtype=np.uint64), np.zeros_like(m, dtype=np.double), 0
            for uid_batch, positives_batch in self.predict_sampler.sample(X):
                batch_size = positives_batch.shape[0]
                hidden = self.model_.init_hidden(batch_size).to(self.device)

                positives_batch = positives_batch.to(self.device)
                scores = torch.zeros((batch_size, self.num_items), device=self.device)
                for i in range(positives_batch.shape[1]): #TODO: it may only be correct if self.bptt == 1, but in the original gru4rec this is the case
                    input_chunk = positives_batch[:,i:i+1]
                    input_mask = (input_chunk != self.pad_token).flatten()
                    # Remove rows with only pad tokens from chunk and from hidden.
                    # We can do this because the array is sorted.
                    true_rows = input_mask
                    true_input_chunk = input_chunk[true_rows]
                    true_hidden = hidden[:, true_rows, :]
                    if true_input_chunk.shape[0] != 0:
                        if i !=0 :
                            target_scores = scores[torch.nonzero(true_rows), true_input_chunk]
                            other_scores = scores[true_rows]
                            ranks = (other_scores > target_scores).sum(axis=1) + 1
                            for j, cutoff in enumerate(m):
                                rec = (ranks <= cutoff).sum()
                                mrr = ((ranks <= cutoff) / ranks).sum()
                                sum_rec[j] += rec.cpu().item()
                                sum_mrr[j] += mrr.cpu().item()
                            measures += len(ranks)
                        output_chunk, hidden[:, true_rows, :] = self.model_(true_input_chunk, true_hidden)
                        item_scores = output_chunk[:, :, :-1].detach().squeeze(dim=1) #.cpu().numpy()
                        scores[true_rows] = item_scores
        for j, cutoff in enumerate(m):
            print(f"Recall@{cutoff}: {sum_rec[j]/measures:.8f} MRR@{cutoff}: {sum_mrr[j]/measures:.8f}")
        return X_pred.tocsr()

class GRU4RecCrossEntropy(GRU4Rec):
    """A recurrent neural network for session-based recommendations.

    The algorithm, also known as GRU4Rec, was introduced in the 2016 and 2018 papers
    "Session-based Recommendations with Recurrent Neural Networks" and
    "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations".

    This version implements the CrossEntropy variant of the algorithm. For negative sampling,
    see :class:`GRU4RecNegSampling`.

    The algorithm makes recommendations by training a recurrent neural network to
    predict the next action of a user, and using the most likely next actions as
    recommendations. At the heart of it is a Gated Recurrent Unit (GRU), a recurrent
    network architecture that is able to form long-term memories.

    Predictions are made by processing a user's actions so far one by one,
    in chronological order::

                                          iid_3_predictions
                                                  |
                 0 --> [ GRU ] --> [ GRU ] --> [ GRU ]
                          |           |           |
                        iid_0       iid_1       iid_2

    here 'iid' are item ids, which can represent page views, purchases, or some other
    action. The GRU builds up a memory of the actions so far and predicts what the
    next action will be based on what other users with similar histories did next.
    While originally devised to make recommendations based on (often short) user
    sessions, the algorithm can be used with long user histories as well.

    For the mathematical details of GRU see "Empirical Evaluation of Gated Recurrent
    Neural Networks on Sequence Modeling" by Chung et al.

    :param num_layers: Number of hidden layers in the RNN. Defaults to 1
    :type num_layers: int, optional
    :param hidden_size: Number of neurons in the hidden layer(s). Defaults to 100
    :type hidden_size: int, optional
    :param num_components: Size of item embeddings. Defaults to 250
    :type num_components: int, optional
    :param dropout: Dropout applied to embeddings and hidden layer(s), 0 for no dropout
    :type dropout: float
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad".
        Defaults to adagrad.
    :type optimization_algorithm: str, optional
    :param momentum: Momentum when using the sgd optimizer
        Defaults to 0.0
    :type momentum: float, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping
        Defaults to 1.0
    :type clipnorm: float, optional
    :param bptt: Number of backpropagation through time steps.
        Defaults to 1
    :type bptt: int, optional
    :param batch_size: Number of examples in a mini-batch.
        Defaults to 512.
    :type batch_size: int, optional
    :param max_epochs: Max training runs through entire dataset.
        Defaults to 5
    :type max_epochs: int, optional
    :param learning_rate: Gradient descent initial learning rate
        Defaults to 0.03
    :type learning_rate: float, optional
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
    :type stopping_criterion: str, optional
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
        Defaults to 0.0
    :type min_improvement: float, optional
    :param seed: Seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If true, the best model will be saved after training,
        defaults to False
    :type save_best_to_file: bool, optional
    :param keep_last: Retain last model, rather than best
        (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param predict_topK: The topK recommendations to keep per row in the matrix.
        Use when the user x item output matrix would become too large for RAM.
        Defaults to None, which results in no filtering.
    :type predict_topK: int, optional
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        num_components: int = 250,
        dropout_p_embed: float = 0,
        dropout_p_hidden: float = 0,
        optimization_algorithm: str = "adagrad",
        momentum: float = 0.0,
        clipnorm: float = 1.0,
        bptt: int = 1,
        batch_size: int = 512,
        max_epochs: int = 5,
        learning_rate: float = 0.03,
        stopping_criterion: str = "recall",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed: int = None,
        save_best_to_file: bool = False,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
        use_correct_weight_init: bool = False,
    ):
        super().__init__(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_components=num_components,
            dropout_p_embed=dropout_p_embed,
            dropout_p_hidden=dropout_p_hidden,
            optimization_algorithm=optimization_algorithm,
            momentum=momentum,
            clipnorm=clipnorm,
            bptt=bptt,
            num_negatives=0,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            stopping_criterion=stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
            use_correct_weight_init=use_correct_weight_init
        )

        self._criterion = nn.CrossEntropyLoss()

    def _compute_loss(
        self,
        output: torch.FloatTensor,
        targets_chunk: torch.LongTensor,
        negatives_chunk: torch.LongTensor,
        true_input_mask: torch.BoolTensor,
    ) -> torch.Tensor:

        return self._criterion(output[true_input_mask], targets_chunk[true_input_mask])


class GRU4RecNegSampling(GRU4Rec):
    """A recurrent neural network for session-based recommendations.

    The algorithm, also known as GRU4Rec, was introduced in the 2016 and 2018 papers
    "Session-based Recommendations with Recurrent Neural Networks" and
    "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations"

    This version implements the Negative Sampling variant of the algorithm. For cross-entropy,
    see :class:`GRU4RecCrossEntropy`.

    The algorithm makes recommendations by training a recurrent neural network to
    predict the next action of a user, and using the most likely next actions as
    recommendations. At the heart of it is a Gated Recurrent Unit (GRU), a recurrent
    network architecture that is able to form long-term memories.

    Predictions are made by processing a user's actions so far one by one,
    in chronological order::

                                          iid_3_predictions
                                                  |
                 0 --> [ GRU ] --> [ GRU ] --> [ GRU ]
                          |           |           |
                        iid_0       iid_1       iid_2

    here 'iid' are item ids, which can represent page views, purchases, or some other
    action. The GRU builds up a memory of the actions so far and predicts what the
    next action will be based on what other users with similar histories did next.
    While originally devised to make recommendations based on (often short) user
    sessions, the algorithm can be used with long user histories as well.

    For the mathematical details of GRU see "Empirical Evaluation of Gated Recurrent
    Neural Networks on Sequence Modeling" by Chung et al.

    Note: Cross-Entropy loss was mentioned in the paper, but omitted for implementation reasons.

    :param num_layers: Number of hidden layers in the RNN. Defaults to 1
    :type num_layers: int, optional
    :param hidden_size: Number of neurons in the hidden layer(s). Defaults to 100
    :type hidden_size: int, optional
    :param num_components: Size of item embeddings. Defaults to 250
    :type num_components: int, optional
    :param dropout: Dropout applied to embeddings and hidden layer(s).
        Defauls to 0 (no dropout)
    :type dropout: float
    :param loss_fn: Loss function. One of "top1", "top1-max", "bpr",
        "bpr-max". Defaults to "bpr"
    :type loss_fn: str, optional
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad".
        Defaults to "adagrad"
    :type optimization_algorithm: str, optional
    :param momentum: Momentum when using the sgd optimizer.
        Defaults to 0.0
    :type momentum: float, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping.
        Defaults to 1.0
    :type clipnorm: float, optional
    :param bptt: Number of backpropagation through time steps.
        Defaults to 1
    :type bptt: int, optional
    :param num_negatives: Number of negatives to sample for every positive.
        Defaults to 50
    :type num_negatives: int, optional
    :param batch_size: Number of examples in a mini-batch.
        Defaults to 512.
    :type batch_size: int, optional
    :param max_epochs: Max training runs through entire dataset.
        Defaults to 5
    :type max_epochs: int, optional
    :param learning_rate: Gradient descent initial learning rate
        Defaults to 0.03
    :type learning_rate: float, optional
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
    :type stopping_criterion: str, optional
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
        Defaults to 0.0
    :type min_improvement: float, optional
    :param seed: Seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If true, the best model will be saved after training,
        defaults to False
    :type save_best_to_file: bool, optional
    :param keep_last: Retain last model, rather than best
        (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param predict_topK: The topK recommendations to keep per row in the matrix.
        Use when the user x item output matrix would become too large for RAM.
        Defaults to None, which results in no filtering.
    :type predict_topK: int, optional
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        num_components: int = 250,
        dropout_p_embed: float = 0,
        dropout_p_hidden: float = 0,
        loss_fn: str = "bpr",
        bpreg: float = 1.0,
        optimization_algorithm: str = "adagrad",
        momentum: float = 0.0,
        clipnorm: float = 1.0,
        bptt: int = 1,
        num_negatives: int = 50,
        batch_size: int = 512,
        max_epochs: int = 5,
        learning_rate: float = 0.03,
        stopping_criterion: str = "recall",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed: int = None,
        save_best_to_file: bool = False,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
        final_activation: str = None,
        use_correct_weight_init: bool = False,
    ):
        super().__init__(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_components=num_components,
            dropout_p_embed=dropout_p_embed,
            dropout_p_hidden=dropout_p_hidden,
            optimization_algorithm=optimization_algorithm,
            momentum=momentum,
            clipnorm=clipnorm,
            bptt=bptt,
            num_negatives=num_negatives,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            stopping_criterion=stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
            final_activation = final_activation,
            use_correct_weight_init = use_correct_weight_init,
        )

        self.loss_fn = loss_fn
        self.bpreg = bpreg

        self._criterion = {
            "top1": top1_loss,
            "top1-max": top1_max_loss,
            "bpr": bpr_loss,
            "bpr-max": bpr_max_loss,
        }[self.loss_fn]

    def _compute_loss(
        self,
        output: torch.FloatTensor,
        targets_chunk: torch.LongTensor,
        negatives_chunk: torch.LongTensor,
        true_input_mask: torch.BoolTensor,
    ) -> torch.Tensor:

        # Select score for positive and negative sample for all tokens
        # For each target, gather the score predicted in output.
        # positive_scores has shape (batch_size x bptt x 1)
        positive_scores = torch.gather(output, 2, targets_chunk.unsqueeze(-1)).squeeze(-1)
        # negative scores has shape (batch_size x bptt x U)
        negative_scores = torch.gather(output, 2, negatives_chunk)

        assert true_input_mask.shape == positive_scores.shape

        true_batch_size, max_hist_len = positive_scores.shape

        # Check if I need to do all this flattening
        true_input_mask_flat = true_input_mask.view(true_batch_size * max_hist_len)

        positive_scores_flat = positive_scores.view(true_batch_size * max_hist_len, 1)
        negative_scores_flat = negative_scores.view(true_batch_size * max_hist_len, negative_scores.shape[2])

        if self.loss_fn == 'bpr-max':
            return self._criterion(positive_scores_flat[true_input_mask_flat], negative_scores_flat[true_input_mask_flat], self.bpreg)
        else:
            return self._criterion(positive_scores_flat[true_input_mask_flat], negative_scores_flat[true_input_mask_flat])


class GRU4RecTorch(nn.Module):
    """PyTorch definition of a basic recurrent neural network for session-based
    recommendations.

    :param num_items: Number of items
    :type num_items: int
    :param hidden_size: Number of neurons in the hidden layer(s)
    :type hidden_size: int
    :param num_components: Size of the item embeddings, None for no embeddings
    :type num_components: int
    :param pad_token: Index of the padding_token
    :type pad_token: int
    :param num_layers: Number of hidden layers, defaults to 1
    :type num_layers: int, optional
    :param dropout: Dropout applied to embeddings and hidden layers, defaults to 0
    :type dropout: float, optional
    """

    def __init__(
        self,
        num_items: int,
        hidden_size: int,
        num_components: int,
        pad_token: int,
        num_layers: int = 1,
        dropout_p_embed: float = 0,
        dropout_p_hidden: float = 0,
        final_activation: str = None,
        use_correct_weight_init: bool = False,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_components = num_components
        self.hidden_size = hidden_size
        self.output_size = num_items
        self.pad_token = pad_token
        self.drop_embed = nn.Dropout(dropout_p_embed)
        self.dropout_p_hidden = dropout_p_hidden
        self.drop_hidden = nn.Dropout(dropout_p_hidden)
        if (final_activation == "linear") or (final_activation is None):
            self.final_activation = nn.Identity()
        elif final_activation.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_activation.split('-')[-1]))
        else:
            raise NotImplementedError
        self.use_correct_weight_init = use_correct_weight_init

        # Passing pad_token will make sure these embeddings are always zero-valued.
        self.emb = nn.Embedding(num_items + 1, num_components, padding_idx=pad_token)
        self.rnn = nn.GRU(
            num_components,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout_p_hidden if num_layers>1 else 0.0,
            batch_first=True,
        )
        # Also return the padding token
        self.lin = nn.Linear(hidden_size, num_items + 1)
        self.init_weights()

        # set the embedding for the padding token to 0
        with torch.no_grad():
            self.emb.weight[pad_token] = torch.zeros(num_components)

    def forward(self, x: torch.LongTensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes scores for each item given an action sequence and previous
        hidden state.

        :param input: Action sequences as a Tensor of shape (A, B)
        :param hidden: Previous hidden state, shape (sequence length, batch size, hidden layer size)
        :return: The computed scores for all items at each point in the sequence
                 for every sequence in the batch, as well as the next hidden state.
                 As Tensors of shapes (A*B, I) and (L, B, H) respectively.
        """
        emb_x = self.emb(x)
        emb_x = self.drop_embed(emb_x)

        # Check if it needs padding
        any_row_requires_padding = (x == self.pad_token).any()

        if any_row_requires_padding:
            seq_lengths = (x != self.pad_token).sum(axis=1)

            padded_emb_x = nn.utils.rnn.pack_padded_sequence(emb_x, seq_lengths.cpu(), batch_first=True)

            padded_rnn_x, hidden = self.rnn(padded_emb_x, hidden)

            rnn_x, _ = nn.utils.rnn.pad_packed_sequence(padded_rnn_x, batch_first=True)

        else:
            rnn_x, hidden = self.rnn(emb_x, hidden)

        if self.dropout_p_hidden > 0:
            rnn_x = self.drop_hidden(rnn_x)
        out = self.lin(rnn_x)
        if self.final_activation != nn.Identity():
            out = self.final_activation(out)

        return out, hidden

    def init_weights(self) -> None:
        """Initializes all model parameters uniformly."""
        if not self.use_correct_weight_init:
            initrange = 0.01
            for param in self.parameters():
                nn.init.uniform_(param, -initrange, initrange)
        else:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if "bias" in name:
                        param.data = torch.zeros_like(param)
                    elif "weight" in name:
                        shape = param.shape
                        sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
                        nn.init.uniform_(param, 0, 2*sigma)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Returns an initial, zero-valued hidden state with shape (L B, H)."""
        return torch.zeros((self.rnn.num_layers, batch_size, self.hidden_size))
