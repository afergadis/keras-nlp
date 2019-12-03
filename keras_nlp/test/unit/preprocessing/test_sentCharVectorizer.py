import numpy as np
from unittest import TestCase
from keras_nlp.preprocessing.text import SentCharVectorizer

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

SMALL_MAX_SENTENCES, SMALL_MAX_WORDS, SMALL_MAX_CHARACTERS = 2, 2, 5
# SWC: SMALL_WORDS_CHARS, TPOST: TRUNCATING POST
DOC0_SWC_TPOST = [[['p', 'l', 'a', 's', 'm'], ['s', 'a', 'm', 'p', 'l']],
                  [['m', 'i'], ['p', 'a', 't', 'i', 'e']]]
DOC1_SWC_TPOST = [[['t', 'h', 'r', 'e', 'e'], ['h', 'u', 'n', 'd', 'r']],
                  [['p', 'a', 'r', 't', 'i'], ['w', 'e', 'r', 'e']]]
# SWC: SMALL_WORDS_CHARS, TPRE: TRUNCATING PRE
DOC0_SWC_TPRE = [[['0'], ['0', '0', '1']],
                 [['o', 'f'], ['m', 'a', 'p', '4', '4']]]
DOC1_SWC_TPRE = [[['o', 'l', 'l', 'o', 'w'], ['u', 'p']],
                 [['3'], ['i', 'v', 'e', 'l', 'y']]]

LARGE_MAX_SENTENCES, LARGE_MAX_WORDS, LARGE_MAX_CHARACTERS = 4, 90, 10


class TestSentenceCharVectorizerWithDefaultValues(TestCase):
    def setUp(self) -> None:
        from nltk import sent_tokenize
        self.sent_tokenize = sent_tokenize
        self.vectorizer = SentCharVectorizer(sent_tokenize)
        self.vectorizer.fit_on_texts(DOCS)

    def test_fit_on_texts(self):
        self.assertEqual(len(self.vectorizer.token2id), 36)

    def test_texts_to_sequences(self):
        vectors = self.vectorizer.texts_to_vectors(DOCS)
        docs_stats, sents_stats, words_stats = self.vectorizer.stats()
        shape = (len(DOCS), docs_stats['max'], sents_stats['max'],
                 words_stats['max'])
        self.assertEqual(vectors.shape, shape)

    def test_vectors_to_texts(self):
        vectors = self.vectorizer.texts_to_vectors(DOCS)
        docs = self.vectorizer.vectors_to_texts(vectors)
        doc_sents = [
            self.vectorizer._apply_filters(sent)
            for sent in self.sent_tokenize(DOC0)
        ]
        sents_words = [sent.split() for sent in doc_sents]
        # The following are the lists of characters of the words of the first
        # sentence of DOC0.
        words_chars = []
        for sent in sents_words:
            words_chars.append([[c for c in word] for word in sent])
        self.assertListEqual(docs[0], words_chars)


class TestSentenceCharVectorizerWithSmallValues(TestCase):
    def setUp(self) -> None:
        from nltk import sent_tokenize
        self.vectorizer = SentCharVectorizer(sent_tokenize)
        self.vectorizer.fit_on_texts(DOCS)

    def test_vectors_to_text_truncating_post(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(SMALL_MAX_SENTENCES, SMALL_MAX_WORDS,
                   SMALL_MAX_CHARACTERS),
            padding='pre',
            truncating='post')
        result = self.vectorizer.vectors_to_texts(vectors)
        expected = [DOC0_SWC_TPOST, DOC1_SWC_TPOST]
        self.assertListEqual(result, expected)

    def test_vectors_to_text_truncating_pre(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(SMALL_MAX_SENTENCES, SMALL_MAX_WORDS,
                   SMALL_MAX_CHARACTERS),
            padding='pre',
            truncating='pre')
        result = self.vectorizer.vectors_to_texts(vectors)
        expected = [DOC0_SWC_TPRE, DOC1_SWC_TPRE]
        self.assertListEqual(result, expected)


class TestSentenceCharVectorizerWithLargeValues(TestCase):
    def setUp(self) -> None:
        from nltk import sent_tokenize
        self.sent_tokenize = sent_tokenize
        self.vectorizer = SentCharVectorizer(sent_tokenize)
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
            shape=(LARGE_MAX_WORDS, LARGE_MAX_CHARACTERS),
            fill_value=self.vectorizer.token2id['PAD'])
        return expected_padded_vector

    def test_text_to_sequences_padding_pre(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(LARGE_MAX_SENTENCES, LARGE_MAX_WORDS,
                   LARGE_MAX_CHARACTERS),
            padding='pre')
        expected_padded_vector = self.expected_vector(DOC0)
        test_vector = vectors[0][0]  # Doc 0, sentence 0
        equiv = np.array_equiv(test_vector, expected_padded_vector)
        self.assertTrue(equiv)

    def test_text_to_sequences_padding_post(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(LARGE_MAX_SENTENCES, LARGE_MAX_WORDS,
                   LARGE_MAX_CHARACTERS),
            padding='post')
        expected_padded_vector = self.expected_vector(DOC0)
        test_vector = vectors[0][-1]  # Doc 0, last sentence
        equiv = np.array_equiv(test_vector, expected_padded_vector)
        self.assertTrue(equiv)


class TestSentenceCharVectorizerExceptions(TestCase):
    def setUp(self) -> None:
        from nltk import sent_tokenize
        self.vectorizer = SentCharVectorizer(sent_tokenize)
        self.vectorizer.fit_on_texts(DOCS)

    def test_text_to_sequences_truncating_exception(self):
        self.assertRaises(
            ValueError, self.vectorizer.texts_to_vectors, DOCS,
            (SMALL_MAX_SENTENCES, SMALL_MAX_WORDS, SMALL_MAX_CHARACTERS),
            'pre', 'middle')

    def test_text__to_sequences_padding_exception(self):
        self.assertRaises(
            ValueError, self.vectorizer.texts_to_vectors, DOCS,
            (SMALL_MAX_SENTENCES, SMALL_MAX_WORDS, SMALL_MAX_CHARACTERS),
            'middle', 'post')
