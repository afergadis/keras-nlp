import numpy as np
from unittest import TestCase

from keras_nlp.metrics.sequence import flatten, strip_pad


class TestUtils(TestCase):
    def setUp(self) -> None:
        self.y = np.array([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]])

    def test_strip_pad(self):
        expected_y = [[1, 1, 2], [1, 1, 2, 2]]
        expected_offset = [2, 1]
        stripped_y, offsets = strip_pad(self.y)
        self.assertListEqual(expected_y, stripped_y)
        self.assertListEqual(expected_offset, offsets)

    def test_flatten(self):
        y_true = self.y[0].reshape(1, -1)
        y_pred = self.y[1].reshape(1, -1)
        y_gold, y_hat = flatten(y_true, y_pred)
        self.assertListEqual([1, 1, 2], y_gold)
        self.assertListEqual([1, 2, 2], y_hat)