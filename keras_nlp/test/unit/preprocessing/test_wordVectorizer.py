from os import path
from collections import Counter
from unittest import TestCase
from keras_nlp.preprocessing import sent_tokenize
from keras_nlp.preprocessing.text import WordVectorizer

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
SMALL_MAX_WORDS = 5
# SWC: SMALL_WORDS, TPOST: TRUNCATING POST
DOC0_SW_TPOST = ['plasma', 'samples', 'were', 'obtained', 'and']
DOC1_SW_TPOST = ['three', 'hundred', 'seventy', 'eight', 'adolescents']
# SWC: SMALL_WORDS_CHARS, TPRE: TRUNCATING PRE
DOC0_SW_TPRE = ['with', 'plasma', 'levels', 'of', 'map44']
DOC1_SW_TPRE = ['1', 'and', '8', '3', 'respectively']

LARGE_MAX_WORDS = 100


class TestWordVectorizerWithDefaultValues(TestCase):
    def setUp(self) -> None:
        self.vectorizer = WordVectorizer()
        self.vectorizer.fit_on_texts(DOCS)

    def test_fit_on_texts(self):
        counts = Counter()
        for doc in DOCS:
            doc = self.vectorizer._apply_filters(doc)
            counts.update(doc.split())
        self.assertEqual(len(self.vectorizer.token2id),
                         len(counts) + 1)  # +PAD

    def test_texts_to_vectors(self):
        vectors = self.vectorizer.texts_to_vectors(DOCS)
        words_stats = self.vectorizer.stats()
        shape = (len(DOCS), words_stats['max'])
        self.assertEqual(vectors.shape, shape)

    def test_vectors_to_texts(self):
        vectors = self.vectorizer.texts_to_vectors(DOCS)
        docs = self.vectorizer.vectors_to_texts(vectors)
        expected_docs = [
            self.vectorizer._apply_filters(doc).split() for doc in DOCS
        ]
        self.assertListEqual(list(docs), expected_docs)


class TestWordVectorizerWithSmallValues(TestCase):
    def setUp(self) -> None:
        self.vectorizer = WordVectorizer()
        self.vectorizer.fit_on_texts(DOCS)

    def test_vectors_to_texts_with_truncating_pre(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS, shape=(SMALL_MAX_WORDS, ), truncating='pre')
        docs = self.vectorizer.vectors_to_texts(vectors)
        self.assertListEqual(docs[0], DOC0_SW_TPRE)
        self.assertListEqual(docs[1], DOC1_SW_TPRE)

    def test_vectors_to_texts_with_truncating_post(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS, shape=(SMALL_MAX_WORDS, ), truncating='post')
        docs = self.vectorizer.vectors_to_texts(vectors)
        self.assertListEqual(docs[0], DOC0_SW_TPOST)
        self.assertListEqual(docs[1], DOC1_SW_TPOST)

    def test_vectors_to_text_truncating_offsets(self):
        vectorizer = WordVectorizer()
        fh = open(path.join(path.dirname(__file__), 'lorem_ipsum.txt'))
        doc = fh.read()
        fh.close()
        doc_sents = sent_tokenize(doc)
        vectorizer.fit_on_texts(doc_sents)
        sents_len = [len(s.split()) for s in doc_sents]  # In words.
        avg = int(sum(sents_len) / len(sents_len))
        target_shape = (avg, )
        truncating_shape = (0.5, 0.5)
        vectors = vectorizer.texts_to_vectors(
            doc_sents, shape=target_shape, truncating=truncating_shape)
        # Don't consider the number of texts.
        self.assertTupleEqual(vectors.shape[1:], target_shape)

        # Check with an even number of words in sentence and 50/50 truncating.
        target_shape = (avg - 1, )
        vectors = vectorizer.texts_to_vectors(
            doc_sents, shape=target_shape, truncating=truncating_shape)
        # Don't consider the number of texts.
        self.assertTupleEqual(vectors.shape[1:], target_shape)


class TestCharVectorizerWithLargeValues(TestCase):
    def setUp(self) -> None:
        self.vectorizer = WordVectorizer()
        self.vectorizer.fit_on_texts(DOCS)

    def test_texts_to_sequences_with_padding_post(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS, shape=(LARGE_MAX_WORDS, ), padding='post')
        # Length of DOC0.
        word_len = len(self.vectorizer._apply_filters(DOC0).split())
        # Number of zero values in the vector.
        num_zeros = LARGE_MAX_WORDS - word_len
        expected_post = [0] * num_zeros
        # Get the last (post) len(num_zeros) values of the DOC0, word0
        post = vectors[0][-num_zeros:].tolist()
        self.assertListEqual(expected_post, post)

    def test_texts_to_vectors_with_padding_pre(self):
        self.vectorizer.fit_on_texts(DOCS)
        vectors = self.vectorizer.texts_to_vectors(
            DOCS, shape=(LARGE_MAX_WORDS, ), padding='pre')
        # Length of DOC0.
        word_len = len(self.vectorizer._apply_filters(DOC0).split())
        # Number of zero values in the vector.
        num_zeros = LARGE_MAX_WORDS - word_len
        expected_post = [0] * num_zeros
        # Get the first (pre) len(num_zeros) values of the DOC0, word0
        post = vectors[0][:num_zeros].tolist()
        self.assertListEqual(expected_post, post)
