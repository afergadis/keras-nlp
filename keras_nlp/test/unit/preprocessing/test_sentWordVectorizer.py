import numpy as np
from unittest import TestCase

from keras_nlp.preprocessing.text import SentWordVectorizer

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

SMALL_MAX_SENTENCES, SMALL_MAX_WORDS = 2, 5
# SWC: SMALL_WORDS, TPOST: TRUNCATING POST
DOC0_SW_TPOST = [['plasma', 'samples', 'were', 'obtained', 'and'],
                 ['mi', 'patients', 'had', '18', 'higher']]
DOC1_SW_TPOST = [['three', 'hundred', 'seventy', 'eight', 'adolescents'],
                 ['participants', 'were', 'assessed', 'at', 'pretest']]
# SWC: SMALL_WORDS_CHARS, TPRE: TRUNCATING PRE
DOC0_SW_TPRE = [['control', 'group', 'p', '0', '001'],
                ['with', 'plasma', 'levels', 'of', 'map44']]
DOC1_SW_TPRE = [['and', '6', 'month', 'follow', 'up'],
                ['1', 'and', '8', '3', 'respectively']]

LARGE_MAX_SENTENCES, LARGE_MAX_WORDS = 5, 100


class TestSentWordVectorizer(TestCase):
    def setUp(self) -> None:
        from nltk import sent_tokenize
        self.sent_tokenize = sent_tokenize
        self.vectorizer = SentWordVectorizer(sent_tokenize)
        self.vectorizer.fit_on_texts(DOCS)

    def test_texts_to_sequences(self):
        vectors = self.vectorizer.texts_to_vectors(DOCS)
        docs_stats, sents_stats, words_stats = self.vectorizer.stats()
        shape = (len(DOCS), docs_stats['max'], sents_stats['max'])
        self.assertEqual(vectors.shape, shape)

    def test_sequences_to_text(self):
        vectors = self.vectorizer.texts_to_vectors(DOCS)
        docs = self.vectorizer.vectors_to_texts(vectors)
        expected_docs = []
        for doc in DOCS:
            _doc = []
            sentences = self.sent_tokenize(doc)
            for sentence in sentences:
                _doc.append(self.vectorizer._apply_filters(sentence).split())
            expected_docs.append(_doc)
        self.assertListEqual(docs, expected_docs)


class TestSentWordVectorizerWithSmallValues(TestCase):
    def setUp(self) -> None:
        from nltk import sent_tokenize
        self.vectorizer = SentWordVectorizer(sent_tokenize)
        self.vectorizer.fit_on_texts(DOCS)

    def test_sequences_to_text_truncating_post(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(SMALL_MAX_SENTENCES, SMALL_MAX_WORDS),
            padding='pre',
            truncating='post')
        result = self.vectorizer.vectors_to_texts(vectors)
        expected = [DOC0_SW_TPOST, DOC1_SW_TPOST]
        self.assertListEqual(result, expected)

    def test_sequences_to_text_truncating_pre(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(SMALL_MAX_SENTENCES, SMALL_MAX_WORDS),
            padding='pre',
            truncating='pre')
        result = self.vectorizer.vectors_to_texts(vectors)
        expected = [DOC0_SW_TPRE, DOC1_SW_TPRE]
        self.assertListEqual(result, expected)


class TestSentWordVectorizerWithLargeValues(TestCase):
    def setUp(self) -> None:
        from nltk import sent_tokenize
        self.sent_tokenize = sent_tokenize
        self.vectorizer = SentWordVectorizer(sent_tokenize)
        self.vectorizer.fit_on_texts(DOCS)

    def expected_vector(self, doc):
        # Length of sentences of DOC0.
        sents = self.sent_tokenize(doc)
        sent_len = len(sents)
        # LARGE_MAX_SENTENCES must be at least by 1 larger from the total
        # sentences of the document, to test padding.
        num_zero_sentences = LARGE_MAX_SENTENCES - sent_len
        if num_zero_sentences < 1:
            self.fail('Unable to test padding. '
                      'Total number of sentences in `vectors` is equal or '
                      'less than `LARGE_MAX_SENTENCES`. '
                      'Increase `LARGE_MAX_SENTENCES`.')
        # Expected padded vector with padding `pre` should have the shape
        # `(LARGE_MAX_WORDS, LARGE_MAX_CHARACTERS)` filled with `pad_value`.
        expected_padded_vector = np.full(
            shape=(LARGE_MAX_SENTENCES, LARGE_MAX_WORDS),
            fill_value=self.vectorizer.token2id['PAD'])
        return expected_padded_vector

    def test_text_to_sequences_padding_pre(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(LARGE_MAX_SENTENCES, LARGE_MAX_WORDS),
            padding='pre')
        expected_padded_vector = self.expected_vector(DOC0)
        test_vector = vectors[0][0]  # Doc 0, sentence 0
        equiv = np.array_equiv(test_vector, expected_padded_vector)
        self.assertTrue(equiv)

    def test_text_to_sequences_padding_post(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(LARGE_MAX_SENTENCES, LARGE_MAX_WORDS),
            padding='post')
        expected_padded_vector = self.expected_vector(DOC0)
        test_vector = vectors[0][-1]  # Doc 0, last sentence
        equiv = np.array_equiv(test_vector, expected_padded_vector)
        self.assertTrue(equiv)
