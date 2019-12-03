import random
import tempfile
import numpy as np
from unittest import TestCase

from keras_nlp.preprocessing import sent_tokenize

from keras_nlp import Dataset, SentCharVectorizer


DOC0 = 'Plasma samples were obtained and analysed with time-resolved ' \
       'immunofluorometric assays determining the plasma levels of MAp44, ' \
       'MASP-1, and MASP-3. ' \
       'MI patients had 18 % higher plasma levels of MAp44 (IQR 11-25 %) as ' \
       'compared to the healthy control group (p < 0.001). ' \
       'However, neither salvage index (Spearman rho -0.1, p = 0.28) nor ' \
       'final infarct size (Spearman rho 0.02, p = 0.83) correlated with ' \
       'plasma levels of MAp44.'
DOC1 = 'Three hundred seventy-eight adolescents ( M age=15.5 years, SD=1.2; ' \
       '68% female, 72% White) with elevated self-assessed depressive ' \
       'symptoms were randomized to a 6-session CB group, minimal contact ' \
       'CB bibliotherapy, or educational brochure control. ' \
       'Participants were assessed at pretest, posttest, and 6-month ' \
       'follow-up. CB group participants showed a significantly lower risk ' \
       'for major depressive disorder onset (0.8%), compared to both CB ' \
       'bibliotherapy (6.3%) and brochure control (6.5%; hazard ratio=8.1 ' \
       'and 8.3, respectively).'
DOCS = [DOC0, DOC1]


def assert_equal(arr1, arr2):
    try:
        np.testing.assert_array_equal(arr1, arr2)
        res = True
    except AssertionError as err:
        res = False
        print(err)
    return res


class TestDataset(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

    def test_save_load_without_vectorizer(self):
        inputs = np.random.rand(10, 5)
        classes = ['y', 'n']
        labels = []
        for i in range(10):
            labels.append(random.choice(classes))
        train_indices = np.array([0, 1, 2, 3, 4, 5])
        dev_indices = np.array([6, 7])
        test_indices = np.array([8, 9])
        ds = Dataset(inputs, labels, train_indices, dev_indices, test_indices)
        ds.save(f'{self.tmp_dir.name}/test.ds')

        # Load ds.
        ds2 = Dataset.load(f'{self.tmp_dir.name}/test.ds')
        self.assertTrue(assert_equal(inputs, ds2.X))
        self.assertTrue(assert_equal(labels, ds2.y))
        self.assertTrue(assert_equal(train_indices, ds2.train_indices))
        self.assertTrue(assert_equal(dev_indices, ds2.dev_indices))
        self.assertTrue(assert_equal(test_indices, ds2.test_indices))

    def test_save_load_with_sent_char_vectorizer(self):
        vectorizer = SentCharVectorizer(sent_tokenize)
        vectorizer.fit_on_texts(DOCS)
        inputs = vectorizer.texts_to_vectors(DOCS)
        labels = [0, 0]
        ds = Dataset(inputs, labels, sent_char_vectorizer=vectorizer)
        ds.save(f'{self.tmp_dir.name}/sent_char_vectorizer.ds')

        ds2 = Dataset.load(f'{self.tmp_dir.name}/sent_char_vectorizer.ds')
        v2docs = vectorizer.vectors_to_texts(inputs)
        scv2docs = ds2.sent_char_vectorizer.vectors_to_texts(inputs)

        self.assertListEqual(v2docs, scv2docs)
