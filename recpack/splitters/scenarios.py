from recpack.splitters.scenario_base import Scenario
import recpack.splitters.splitter_base as splitter_base
from recpack.utils import get_logger


class StrongGeneralization(Scenario):
    def __init__(self, perc_users_train: float, perc_interactions_in: float, validation=False):
        """
        Strong generalization splits your data so that a user can only be in one of
        training, validation or test.
        This is considered more robust, harder and more realistic than weak generalization.

        :param perc_users_train: Percentage of users used for training, between 0 and 1.
        :type perc_users_train: float
        :param perc_interactions_in: Percentage of the user's interactions to be used as history.
        :type perc_interactions_in: float
        :param validation: Split
        :type validation: boolean, optional
        """
        super().__init__()
        self.perc_users_train = perc_users_train
        self.validation = validation
        self.perc_interactions_in = perc_interactions_in

        self.strong_gen = splitter_base.StrongGeneralizationSplitter(perc_users_train)
        self.interaction_split = splitter_base.PercentageInteractionSplitter(perc_interactions_in)

    def split(self, data):
        self.training_data, val_te_data = self.strong_gen.split(data)

        if self.validation:
            val_data, te_data = self.validation_splitter.split(val_te_data)
            self.validation_data_in, self.validation_data_out = self.interaction_split.split(val_data)
        else:
            te_data = val_te_data

        self.test_data_in, self.test_data_out = self.interaction_split.split(te_data)


class TrainingInTestOutTimed(Scenario):

    def __init__(self, t, t_delta=None, t_alpha=None, validation=False):
        """
        Training in, test out (timed) splits your data so that a user's history is split in time,
        where only data before t can be used to train a model, and is later used to make
        recommendations for that user.
        This scenario resembles reality most closely.

        :param t: Time (as seconds since epoch) to split along.
        :type t: int
        :param t_delta: If t_delta is specified, use only interactions in the interval [t, t_delta) for test and validation.
        :type t_delta: int, optional
        :param t_alpha: If t_alpha is specified, use only interactions in the interval [t_alpha, t) for training.
        :type t_alpha: int, optional
        :param validation: Split
        :type validation: boolean, optional
        """
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha
        self.validation = validation
        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

    def split(self, data):
        self.training_data, val_te_data_out = self.timestamp_spl.split(data)

        if self.validation:
            self.validation_data_in = self.training_data
            self.validation_data_out, self.test_data_out = self.validation_splitter.split(val_te_data_out)
        else:
            self.test_data_out = val_te_data_out

        self.test_data_in = self.training_data


class StrongGeneralizationTimed(Scenario):
    """
    Split data in strong generalization fashion, also splitting on timestamp.
    training_data = user in set of sampled test users + event time < t
    test_data_in = user not in the set of test users + event_time < t
    test_data_out = user not in the set of test users + event_time > t
    """
    # TODO Make this more standard.

    def __init__(self, perc_users_in, t, t_delta=None, t_alpha=None, validation=False):
        super().__init__()
        self.perc_users_in = perc_users_in
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha
        self.validation = validation

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)
        self.strong_gen = splitter_base.StrongGeneralizationSplitter(perc_users_in)

    def split(self, data):
        # Split across user dimension
        tr_data, val_te_data = self.strong_gen.split(data)
        # Make sure only interactions before t are used for training
        self.training_data, _ = self.timestamp_spl.split(tr_data)

        if self.validation:
            val_data, te_data = self.validation_splitter.split(val_te_data)
            self.validation_data_in, self.validation_data_out = self.timestamp_spl.split(val_data)
        else:
            te_data = val_te_data

        self.test_data_in, self.test_data_out = self.timestamp_spl.split(te_data)


class TimedOutOfDomainPredictAndEvaluate(Scenario):
    """
    Class to split data from 2 input datatypes, one of the data types will be used as training data,
    the other one will be used as test data.
    An example use case is training a model on pageviews,
    and evaluating with purchases as test_in, test_out and split on time

    training_data = data_1 & timestamp < t
    test_data_in = data_2 & timestamp < t
    test_data_out = data_2 & timestamp > t
    """
    # TODO Make this more standard.

    def __init__(self, t, t_alpha=None, t_delta=None, validation=False):
        super().__init__()
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha
        self.validation = validation

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

    def split(self, data, data_2):
        self.training_data, _ = self.timestamp_spl.split(data)

        if self.validation:
            val_data, te_data = self.validation_splitter.split(data_2)
            self.validation_data_in, self.validation_data_out = self.timestamp_spl.split(val_data)
        else:
            te_data = data_2

        self.test_data_in, self.test_data_out = self.timestamp_spl.split(te_data)


class TrainInTimedOutOfDomainEvaluate(Scenario):
    """
    Class to split two input datatypes for training and testing.
    train_data = data_1 & timestamp < t
    test_in = train_data
    test_out = data_2 & timestamp > t
    """
    # TODO Make this more standard.

    def __init__(self, t, t_alpha=None, t_delta=None, validation=False):
        super().__init__()
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha
        self.validation = validation

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

    def split(self, data, data_2):
        self.training_data, _ = self.timestamp_spl.split(data)

        if self.validation:
            val_data, te_data = self.validation_splitter.split(data_2)
            _, self.validation_data_out = self.timestamp_spl.split(val_data)
            self.validation_data_in = self.training_data
        else:
            te_data = data_2

        _, self.test_data_out = self.timestamp_spl.split(te_data)
        self.test_data_in = self.training_data
