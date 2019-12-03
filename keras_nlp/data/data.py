import logging
import os

import numpy as np
from keras_nlp import utils


class Dataset(object):
    def __init__(self,
                 inputs,
                 labels,
                 train_indices=None,
                 dev_indices=None,
                 test_indices=None,
                 **kwargs):
        """
        Encapsulates all pieces of preprocessed data to run an experiment.

        This is basically a bag of items that makes it easy to serialize and
        deserialize everything as a unit.

        Parameters
        ----------
        inputs : array like
            The vectorized texts. This input will be the input of the Keras
            model.

        labels : array like, list
            The numerical values of the labels.

        train_indices, dev_indices, test_indices : array like or list, default=None, optional
            The optional indices to use for each data split.

        **kwargs
            Additional key value items to store. It is handy to store the
            vectorizer(s) used to prepare inputs and the labels names.
        """
        self.X = np.array(inputs)
        self.y = np.array(labels)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._train_indices = train_indices
        self._dev_indices = dev_indices
        self._test_indices = test_indices

    def save(self, file_path):
        """Serializes this dataset to a file.

        Parameters
        ----------
        file_path : str
            The file path to use.

        """
        logging.info(f'Saving to "{file_path}"')
        utils.dump(self, file_path)

    @staticmethod
    def load(file_path):
        """Loads the dataset from a file.

        Parameters
        ----------
        file_path : str
            The file path to use.

        Returns
        -------
        `Dataset`
            The Dataset instance.
        """
        if os.path.exists(file_path):
            logging.info(f'Loading from "{file_path}"')
        return utils.load(file_path)

    @property
    def train_indices(self):
        """Get the train indices. """
        return self._train_indices

    @property
    def dev_indices(self):
        """Get the train indices. """
        return self._dev_indices

    @property
    def test_indices(self):
        """Get the test indices. """
        return self._test_indices

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in state:
            del state['logger']
        return state
