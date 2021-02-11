import numpy as np
import os
import pandas as pd
from urllib.request import urlretrieve
import zipfile

from recpack.preprocessing.filters import Filter, MinItemsPerUser, MinUsersPerItem
from recpack.data.matrix import InteractionMatrix, to_binary
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.util import to_tuple


def _fetch_remote(url, filename):
    """Fetch data from remote url and save locally

    :param url: url to fetch data from
    :type url: str
    :param filename: Path to save file to
    :type filename: str
    :return: The filename where data was saved
    :rtype: str
    """
    # TODO: checksum?
    urlretrieve(url, filename)
    return filename


class Dataset:
    """Handle public datasets.

    If available at a public url the dataset will downloaded if not present
    :param filename: Where to look for the file with data.
    :type filename: str
    :param preprocess_default: Should a default set of filters be initialised? Defaults to True
    :type preprocess_default: bool, optional
    """

    USER_IX = "user_id"
    ITEM_IX = "item_id"
    TIMESTAMP_IX = "seconds_since_epoch"

    def __init__(self, filename, preprocess_default=True):
        self.filename = filename
        self.preprocessor = DataFramePreprocessor(
            self.ITEM_IX, self.USER_IX, self.TIMESTAMP_IX, dedupe=False
        )
        if preprocess_default:
            for f in self._default_filters:
                self.add_filter(f)

    @property
    def _default_filters(self):
        # Return a list of filters to be added to the preprocessor
        # if default is enabled.

        return [
            MinItemsPerUser(3, self.ITEM_IX, self.USER_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]

    def add_filter(self, _filter: Filter):
        """Add a filter to be used when calling load_interaction_matrix()

        :param _filter: Filter to be applied to the laoded dataframe
                    processing to interaction matrix.
        :type _filter: Filter
        """
        self.preprocessor.add_filter(_filter)

    def fetch_dataset(self, force=False):
        """Check if dataset is present, if not download

        :param force: If True, dataset will be downloaded,
                even if the file already exists.
                Defaults to False.
        :type force: bool, optional
        """
        if not os.path.exists(self.filename) or force:
            self._download_dataset()

    def _download_dataset(self):
        raise NotImplementedError("Should still be implemented")

    def load_dataframe(self):
        raise NotImplementedError("Needs to be implemented")

    def load_interaction_matrix(self):
        """Parse the loaded dataframe into an InteractionMatrix.

        During parsing the added filters are applied in order.
        :return: The resulting InteractionMatrix
        :rtype: InteractionMatrix
        """
        df = self.load_dataframe()

        return self.preprocessor.process(df)


class CiteULike(Dataset):
    """Dataset class for the CiteULike dataset.

    Full information  on the dataset can be found at https://github.com/js05212/citeulike-a.
    Uses the `users.dat` file from the dataset to construct an implicit feedback interaction matrix.

    Default processing makes sure that:
    - Each remaining user has interacted with at least 3 items
    - Each remaining  item has been interacted with by at least 5 users.dat

    :param filename: Where to look for the file with data.
    :type filename: str
    :param preprocess_default: Should a default set of filters be initialised? Defaults to True
    :type preprocess_default: bool, optional
    """

    TIMESTAMP_IX = None

    def _download_dataset(self):
        """Download thee users.dat file from the github repository.
        The file is saved at the specified `self.filename`
        """
        DATASETURL = (
            "https://raw.githubusercontent.com/js05212/citeulike-a/master/users.dat"
        )
        _fetch_remote(DATASETURL, self.filename)

    def load_dataframe(self):
        """Load the data from file, and return a dataframe.

        Downloads the data file if it is not yet present.
        The output will contain a dataframe with a user_id and item_id column.
        Each interaction is stored in a separate row.

        :return: The interactions as a dataframe, with a row for each interaction.
        :rtype: pd.DataFrame
        """

        # This will download the dataset if necessary
        self.fetch_dataset()

        u_i_pairs = []
        with open(self.filename, "r") as f:

            for user, line in enumerate(f.readlines()):
                item_cnt = line.strip("\n").split(" ")[0]  # First element is a count
                items = line.strip("\n").split(" ")[1:]
                assert len(items) == int(item_cnt)

                for item in items:
                    assert item.isdecimal()  # Make sure the identifiers are correct.
                    u_i_pairs.append((user, int(item)))

        # Rename columns to default ones ?
        df = pd.DataFrame(u_i_pairs, columns=[self.USER_IX, self.ITEM_IX])
        return df


class MovieLens25M(Dataset):
    """Handles Movielens 25M dataset.

    All information on the dataset can be found at https://grouplens.org/datasets/movielens/25m/.
    Uses the `ratings.csv` file to generate an interaction matrix.

    Default processing makes sure that:
    - Each remaining user has interacted with at least 3 items
    - Each remaining  item has been interacted with by at least 5 users

    To only use ratings above a certain value as interactions, you have to add this filter.
    You want this filter to be applied before the default ones, so you can use:

    ```
    from recpack.preprocessing.filters import MinRating
    from recpack.data.datasets import MovieLens25M
    d = MovieLens25M('path/to/file')
    d.add_filter(MinRating("rating", 3), index=0)
    ```
    """

    USER_IX = "userId"
    ITEM_IX = "movieId"
    TIMESTAMP_IX = "timestamp"
    RATING_IX = "rating"

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.filename`
        """
        DATASETURL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"

        # Get the director part of the file specified
        dir_f = os.path.dirname(self.filename)

        # Download the zip into the directory
        _fetch_remote(DATASETURL, os.path.join(dir_f, "ml-25m.zip"))

        # Extract the ratings file which we will use
        with zipfile.ZipFile(os.path.join(dir_f, "ml-25m.zip"), "r") as zip_ref:
            zip_ref.extract("ml-25m/ratings.csv", dir_f)

        # Rename the ratings file to the specified filename
        os.rename(os.path.join(dir_f, "ml-25m/ratings.csv"), self.filename)

    def load_dataframe(self):
        """Load the data from file, and return a dataframe.

        Downloads the data file if it is not yet present.
        The output will contain the data of the CSV file, in Dataframe format.

        :return: The interactions as a dataframe, with a row for each interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()
        df = pd.read_csv(self.filename)

        return df


class RecsysChallenge2015(Dataset):
    """Handles data from the Recsys Challenge 2015, yoochoose dataset.

    All information and downloads can be found at https://www.kaggle.com/chadgostopp/recsys-challenge-2015.
    Because downloading the data requires a Kaggle account we can't download it here,
    you should download the data manually and provide the path to the `yoochoose-clicks.dat` file.



    Default processing makes sure that:
    - Each remaining user has interacted with at least 3 items
    - Each remaining  item has been interacted with by at least 5 users.dat
    """

    USER_IX = "session"

    def _download_dataset(self):
        """Downloading this dataset is not supported."""
        raise NotImplementedError(
            "Recsys Challenge dataset should be downloaded manually, you can get it at: https://www.kaggle.com/chadgostopp/recsys-challenge-2015."
        )

    @property
    def _default_filters(self):
        # Return a list of filters to be added to the preprocessor
        # if default is enabled.
        # TODO: this had a different preprocessing in the function compared to the others.
        # Do we try to unify this, and make all defaults the same,
        # or was there a particular reason for this one?
        return [
            MinItemsPerUser(3, self.USER_IX, self.ITEM_IX, count_duplicates=True),
            MinUsersPerItem(5, self.USER_IX, self.ITEM_IX, count_duplicates=True),
            MinItemsPerUser(3, self.USER_IX, self.ITEM_IX, count_duplicates=True),
        ]

    def load_dataframe(self):
        """Load the data from file, and return a dataframe.

        The output will contain a dataframe with a user_id, item_id and seconds_since_epoch column.
        Each interaction is stored in a separate row.

        :return: The interactions as a dataframe, with a row for each interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()

        df = pd.read_csv(
            self.filename,
            names=[self.USER_IX, self.TIMESTAMP_IX, self.ITEM_IX],
            dtype={
                self.USER_IX: np.int64,
                self.TIMESTAMP_IX: np.str,
                self.ITEM_IX: np.int64,
            },
            parse_dates=[self.TIMESTAMP_IX],
            usecols=[0, 1, 2],
            # nrows=nrows,
        )

        # Adapt timestamp, this makes it so the timestamp is always seconds since epoch
        df[self.TIMESTAMP_IX] = (
            df[self.TIMESTAMP_IX].astype(int) / 1e9
        )  # pandas datetime -> seconds from epoch

        return df
