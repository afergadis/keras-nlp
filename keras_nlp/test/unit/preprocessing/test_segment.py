from unittest import TestCase

from keras_nlp.preprocessing.segment import SentenceSplitter

doc = """MI patients had 18% higher plasma levels of MAp44 (IQR 11-25%) as 
compared to the healthy control group (p < 0.001). However, neither salvage 
index (Spearman rho -0.1, p = 0.28) nor final infarct size 
(Spearman rho 0.02, p = 0.83) correlated with plasma levels of MAp44. 
"""
sentences = [
    "MI patients had 18% higher plasma levels of MAp44 (IQR 11-25%) as "
    "compared to the healthy control group (p < 0.001).",
    "However, neither salvage index (Spearman rho -0.1, p = 0.28) nor final "
    "infarct size (Spearman rho 0.02, p = 0.83) correlated with plasma "
    "levels of MAp44."
]


class TestSegment(TestCase):
    def test_tokenize(self):
        ss = SentenceSplitter()
        tokenized = ss.tokenize(doc)
        self.assertListEqual(tokenized, sentences)
