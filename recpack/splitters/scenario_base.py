from abc import ABC, abstractmethod
from typing import Tuple, Union
from warnings import warn

import recpack.splitters.splitter_base as splitter_base
from recpack.data.matrix import InteractionMatrix


class Scenario(ABC):
    """Base class for defining an evaluation scenario.

    A scenario is a set of steps that splits data into training,
    validation and test datasets.
    Both validation and test dataset are made up of two components:
    a fold-in set of interactions that is used to predict another held-out
    set of interactions.

    :param validation: Create a validation dataset when True, else split into training and test datasets.
    :type validation: boolean, optional
    """

    def __init__(self, validation=False):
        self.validation = validation
        if validation:
            self.validation_splitter = splitter_base.StrongGeneralizationSplitter(
                0.8)

    @abstractmethod
    def _split(self, data_m: InteractionMatrix) -> None:
        """Abstract method to be implemented by the scenarios.

        Splits the data and assigns to :attr:`self.train_X`,
        :attr:`self.test_data_in, :attr:`self.test_data_out`, :attr:`self.validation_data_in`
        and :attr:`self.validation_data_out`

        :param data_m: Interaction matrix to be split.
        :type data_m: InteractionMatrix
        """

    def split(self, data_m: InteractionMatrix) -> None:
        """Splits ``data_m`` according to the scenario.

        After splitting properties :attr:`training_data`,
        :attr:`validation_data` and :attr:`test_data` can be used to retrieve the splitted data.

        :param data_m: Interaction matrix that should be split.
        :type data: InteractionMatrix
        """
        self._split(data_m)

        self._check_split()

    @property
    def training_data(self) -> InteractionMatrix:
        """The training dataset.

        :return: Interaction Matrix of training interactions.
        :rtype: InteractionMatrix
        """
        if not hasattr(self, "train_X"):
            raise KeyError("Split before trying to access the training_data property.")

        return self.train_X

    @property
    def validation_data(
        self,
    ) -> Union[Tuple[InteractionMatrix, InteractionMatrix], None]:
        """The validation dataset. Consist of a fold-in and hold-out set of interactions.

        Data is processed such that both matrices contain the exact same users.
        Users that were present in only one of the matrices and not in the other are removed.

        :return: Validation data matrices as
            InteractionMatrix in, InteractionMatrix out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        if not hasattr(self, "_validation_data_in"):
            raise KeyError("Split before trying to access the validation_data property.")

        if not self.validation:
            raise KeyError("This scenario was created without validation_data.")

        # make sure users match both.
        in_users = self._validation_data_in.active_users
        out_users = self._validation_data_out.active_users

        matching = list(in_users.intersection(out_users))
        return (
            self._validation_data_in.users_in(matching),
            self._validation_data_out.users_in(matching),
        )

    @property
    def validation_data_in(self):
        return self.validation_data[0]

    @property
    def validation_data_out(self):
        return self.validation_data[1]

    @property
    def test_data_in(self):
        return self.test_data[0]

    @property
    def test_data_out(self):
        return self.test_data[1]

    @property
    def test_data(self) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """The test dataset. Consist of a fold-in and hold-out set of interactions.

        Data is processed such that both matrices contain the exact same users.
        Users that were present in only one of the matrices and not in the other are removed.

        :return: Test data matrices as InteractionMatrix in, InteractionMatrix out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        # make sure users match.
        in_users = self._test_data_in.active_users
        out_users = self._test_data_out.active_users

        matching = list(in_users.intersection(out_users))
        return (
            self._test_data_in.users_in(matching),
            self._test_data_out.users_in(matching),
        )

    def _check_split(self):
        """Checks that the splits have been done properly.

        Makes sure all expected attributes are set.
        """
        assert hasattr(self, "train_X") and self.train_X is not None
        if self.validation:
            assert (
                hasattr(self, "_validation_data_in")
                and self._validation_data_in is not None
            )
            assert (
                hasattr(self, "_validation_data_out")
                and self._validation_data_out is not None
            )

        assert hasattr(self, "_test_data_in") and self._test_data_in is not None
        assert hasattr(
            self, "_test_data_out") and self._test_data_out is not None

        self._check_size()

    def _check_size(self):
        """
        Warns user if any of the sets is unusually small or empty
        """
        n_train = self.train_X.num_interactions
        n_test_in = self._test_data_in.num_interactions
        n_test_out = self._test_data_out.num_interactions
        n_test = n_test_in + n_test_out
        n_total = n_train + n_test

        if self.validation:
            n_val_in = self._validation_data_in.num_interactions
            n_val_out = self._validation_data_out.num_interactions
            n_val = n_val_in + n_val_out
            n_total += n_val

        def check(name, count, total, threshold):
            if (count + 1e-9) / (total + 1e-9) < threshold:
                warn(
                    f"{name} resulting from {type(self).__name__} is unusually small.")

        check("Training set", n_train, n_total, 0.05)
        check("Test set", n_test, n_total, 0.01)
        check("Test in set", n_test_in, n_test, 0.05)
        check("Test out set", n_test_out, n_test, 0.01)
        if self.validation:
            check("Validation set", n_val, n_total, 0.01)
            check("Validation in set", n_val_in, n_val, 0.05)
            check("Validation out set", n_val_out, n_val, 0.01)
