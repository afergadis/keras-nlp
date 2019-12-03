import tempfile
from unittest import TestCase
from keras_nlp import Glove, WordVectorizer, W2V


class TestGlove(TestCase):
    def setUp(self) -> None:
        self.vectors_file = tempfile.NamedTemporaryFile()
        self.vectors_file.write(b'phasellus 0.1 -0.3 0.2\n')
        self.vectors_file.write(b'fermentum 0.2 0.1 -0.1\n')
        self.vectors_file.seek(0)
        texts = [
            'Phasellus fermentum tellus eget libero sodales varius.',
            'In vestibulum erat nec nulla porttitor.'
        ]
        self.word_vectorizer = WordVectorizer(oov_token='_UNK_', verbose=0)
        self.word_vectorizer.fit_on_texts(texts)

    def test_glove_get_embedding_layer(self):
        glove = Glove(self.word_vectorizer.token2id,
                      self.word_vectorizer.oov_token)
        glove.load(self.vectors_file.name)
        embedding_layer = glove.get_embedding_layer(input_length=7)
        self.assertEqual(embedding_layer.input_dim,
                         self.word_vectorizer.num_tokens)


class TestW2V(TestCase):
    def setUp(self) -> None:
        self.vectors_file = tempfile.NamedTemporaryFile()
        self.vectors_file.write(b'2 3\n')
        self.vectors_file.write(b'phasellus 0.1 -0.3 0.2\n')
        self.vectors_file.write(b'fermentum 0.2 0.1 -0.1\n')
        self.vectors_file.seek(0)
        texts = [
            'Phasellus fermentum tellus eget libero sodales varius.',
            'In vestibulum erat nec nulla porttitor.'
        ]
        self.word_vectorizer = WordVectorizer(oov_token='_UNK_', verbose=0)
        self.word_vectorizer.fit_on_texts(texts)

    def test_w2v_get_embedding_layer(self):
        w2v = W2V(self.word_vectorizer.token2id,
                  self.word_vectorizer.oov_token)
        w2v.load(self.vectors_file.name)
        embedding_layer = w2v.get_embedding_layer(input_length=7)
        self.assertEqual(embedding_layer.input_dim,
                         self.word_vectorizer.num_tokens)
