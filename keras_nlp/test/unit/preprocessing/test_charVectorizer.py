from unittest import TestCase
from os import path
from keras_nlp.preprocessing import sent_tokenize
from keras_nlp.preprocessing.text import CharVectorizer

DOC0 = 'Plasma samples were obtained and analysed with time-resolved ' \
       'immunofluorometric assays determining the plasma levels of MAp44, ' \
       'MASP-1, and MASP-3. ' \
       'MI patients had 18 % higher plasma levels of MAp44 (IQR 11-25 %) as ' \
       'compared to the healthy control group (p < 0.001). ' \
       'However, neither salvage index (Spearman rho -0.1, p = 0.28) nor '\
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
DOC0_SWC_TPOST = [['p', 'l', 'a', 's', 'm'], ['s', 'a', 'm', 'p', 'l']]
DOC1_SWC_TPOST = [['t', 'h', 'r', 'e', 'e'], ['h', 'u', 'n', 'd', 'r']]
# SWC: SMALL_WORDS_CHARS, TPRE: TRUNCATING PRE
DOC0_SWC_TPRE = [['o', 'f'], ['m', 'a', 'p', '4', '4']]
DOC1_SWC_TPRE = [['3'], ['i', 'v', 'e', 'l', 'y']]

LARGE_MAX_SENTENCES, LARGE_MAX_WORDS, LARGE_MAX_CHARACTERS = 4, 90, 20


class TestCharVectorizerWithDefaultValues(TestCase):
    def setUp(self) -> None:
        self.vectorizer = CharVectorizer()
        self.vectorizer.fit_on_texts(DOCS)

    def test_fit_on_texts(self):
        self.assertEqual(len(self.vectorizer.token2id), 36)

    def test_texts_to_vectors(self):
        vectors = self.vectorizer.texts_to_vectors(DOCS)
        docs_stats, words_stats = self.vectorizer.stats()
        shape = (len(DOCS), docs_stats['max'], words_stats['max'])
        self.assertEqual(vectors.shape, shape)

    def test_vectors_to_texts(self):
        vectors = self.vectorizer.texts_to_vectors(DOCS)
        docs_chars = self.vectorizer.vectors_to_texts(vectors)
        text1 = self.vectorizer._apply_filters(DOC0)
        doc1_chars = [[c for c in word] for word in text1.split()]
        self.assertListEqual(docs_chars[0], doc1_chars)

        docs_words = self.vectorizer.vectors_to_texts(vectors, as_words=True)
        doc1_words = [word for word in text1.split()]
        self.assertListEqual(docs_words[0], doc1_words)


class TestCharVectorizerWithSmallValues(TestCase):
    def setUp(self) -> None:
        self.vectorizer = CharVectorizer()
        self.vectorizer.fit_on_texts(DOCS)

    def test_vectors_to_texts_with_truncating_pre(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(SMALL_MAX_WORDS, SMALL_MAX_CHARACTERS),
            truncating='pre')
        docs = self.vectorizer.vectors_to_texts(vectors)
        self.assertListEqual(docs[0], DOC0_SWC_TPRE)
        self.assertListEqual(docs[1], DOC1_SWC_TPRE)

    def test_vectors_to_texts_with_truncating_post(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(SMALL_MAX_WORDS, SMALL_MAX_CHARACTERS),
            truncating='post')
        docs = self.vectorizer.vectors_to_texts(vectors)
        self.assertListEqual(docs[0], DOC0_SWC_TPOST)
        self.assertListEqual(docs[1], DOC1_SWC_TPOST)

    def test_vectors_to_text_truncating_offsets(self):
        vectorizer = CharVectorizer()
        doc = open(path.join(path.dirname(__file__), 'lorem_ipsum.txt')).read()
        doc_sents = sent_tokenize(doc)
        vectorizer.fit_on_texts(doc_sents)
        doc_sents = [
            vectorizer._apply_filters(s) for s in doc_sents
        ]
        words_len = [len(sent.split()) for sent in doc_sents]
        chars_len = [len(word) for sent in doc_sents for word in sent.split()]
        avg_words = int(sum(words_len) / len(words_len))
        avg_chars = int(sum(chars_len) / len(chars_len))
        target_shape = (avg_words, avg_chars)
        truncating_shape = (0.5, 0.5)
        vectors = vectorizer.texts_to_vectors([doc],
                                              shape=target_shape,
                                              truncating=truncating_shape)
        # Don't consider the number of texts.
        self.assertTupleEqual(vectors.shape[1:], target_shape)


class TestCharVectorizerWithLargeValues(TestCase):
    def setUp(self) -> None:
        self.vectorizer = CharVectorizer()
        self.vectorizer.fit_on_texts(DOCS)

    def test_texts_to_sequences_with_padding_post(self):
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(LARGE_MAX_WORDS, LARGE_MAX_CHARACTERS),
            padding='post')
        # Length of 1st word of DOC0.
        word_len = len(DOC0.split()[0])
        # Number of zero values in the vector.
        num_zeros = LARGE_MAX_CHARACTERS - word_len
        expected_post = [0] * num_zeros
        # Get the last (post) len(num_zeros) values of the DOC0, word0
        post = vectors[0][0][-num_zeros:].tolist()
        self.assertListEqual(expected_post, post)

    def test_texts_to_sequences_with_padding_pre(self):
        self.vectorizer.fit_on_texts(DOCS)
        vectors = self.vectorizer.texts_to_vectors(
            DOCS,
            shape=(LARGE_MAX_WORDS, LARGE_MAX_CHARACTERS),
            padding='pre')
        # Length of 1st word of DOC0.
        word_len = len(DOC0.split()[0])
        # Number of zero values in the vector.
        num_zeros = LARGE_MAX_CHARACTERS - word_len
        expected_post = [0] * num_zeros
        # Get the first (pre) len(num_zeros) values of the DOC0, word0
        post = vectors[0][0][:num_zeros].tolist()
        self.assertListEqual(expected_post, post)


class TestCharVectorizerWithNumChars(TestCase):
    def setUp(self) -> None:
        self.num_chars = 20
        self.vectorizer = CharVectorizer(num_chars=self.num_chars)

    def test_fit_on_texts(self):
        self.vectorizer.fit_on_texts(DOCS)
        self.assertEqual(len(self.vectorizer.token2id),
                         self.num_chars + 1)  # Add PAD.


class TestCharVectorizerWithCharactersSet(TestCase):
    def setUp(self) -> None:
        self.characters = [c for c in 'aeiouy']
        self.vectorizer = CharVectorizer(characters=self.characters)

    def test_fit_on_texts(self):
        self.vectorizer.fit_on_texts(DOCS)
        self.assertEqual(
            len(self.vectorizer.token2id),
            len(self.characters) + 1)  # Add PAD.


class TestCharVectorizerExceptions(TestCase):
    def setUp(self) -> None:
        self.vectorizer = CharVectorizer()
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
