import abc
import struct
import logging
import numpy as np
from keras.utils.generic_utils import Progbar
from keras.layers import Embedding


class WordVectors(abc.ABC):
    """
    A class to define common methods and functionality to load word vectors
    from different formats.

    Examples
    --------
    >>> import tempfile
    >>> from keras_nlp import Glove, WordVectorizer
    >>> vectors_file = tempfile.NamedTemporaryFile()
    >>> vectors_file.write(b'phasellus 0.1 -0.3 0.2\\n')  #doctest:+SKIP
    >>> vectors_file.write(b'fermentum 0.2 0.1 -0.1\\n')  #doctest:+SKIP
    >>> vectors_file.seek(0)  #doctest:+SKIP
    >>> texts = ['Phasellus fermentum tellus eget libero sodales varius.',
    ... 'In vestibulum erat nec nulla porttitor.']
    >>> word_vectorizer = WordVectorizer(oov_token='_UNK_', verbose=0)
    >>> word_vectorizer.fit_on_texts(texts)
    >>> glove = Glove(word_vectorizer.token2id, word_vectorizer.oov_token)
    >>> glove.load(vectors_file.name)

    Validate the loaded vectors.
    >>> arr1 = glove['phasellus']
    >>> arr2 = np.asarray([0.1, -0.3, 0.2])
    >>> np.testing.assert_array_almost_equal(arr1, arr2, decimal=1)

    Embedding layer input dim should match vocabulary number of tokens.
    >>> embedding_layer = glove.get_embedding_layer(input_length=7)
    >>> assert embedding_layer.input_dim = word_vectorizer.num_tokens
    """

    def __init__(self, vocab, oov_token=None):
        """
        Parameters
        ----------
        vocab : dict
            A token to id mapping.

        oov_token : str or None, default None
            If the vocabulary has an out-of-vocabulary token, then this
            parameter should have the key of that token. A random vector is
            create with uniform distribution in range [-0.05, 0.05].
        """
        self.vocab = vocab
        self.oov_token = oov_token
        self.vector_len = 0
        self.vectors = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def _init_vectors(self):
        """
        Initialize vectors array and insert a random vector in case of having
        an out-of-vocabulary token.
        """
        self.vectors = np.zeros(shape=(len(self.vocab), self.vector_len))
        oov_id = None
        if self.oov_token is not None:
            oov_id = self.vocab[self.oov_token]
            self.vectors[oov_id] = np.random.uniform(-0.05, 0.05,
                                                     self.vector_len)
        return oov_id

    def _replace_oov_tokens(self, oov_token_id):
        """
        Replace the vectors of words in vocab not found in the loaded file,
        with the vector of the oov token.

        Parameters
        ----------
        oov_token_id : int
            The index in vectors to the vector of the oov token.
        """
        zeros_vector = np.zeros(shape=(self.vectors.shape[-1],))
        oov_vector = self.vectors[oov_token_id]
        for word, idx in self.vocab.items():
            if idx == 0:
                continue
            if np.array_equal(self.vectors[idx], zeros_vector):
                self.vectors[idx] = oov_vector

    @abc.abstractmethod
    def load(self, file_path):
        """
        Read word vectors.

        Parameters
        ----------
        file_path : str
            File name to the glove file.
        """
        raise NotImplementedError

    def get_embedding_layer(self, input_length, output_dim=None, **kwargs):
        """
        Builds an Embedding layer.

        If we have loaded word vectors from a file, then those vectors will
        be used as weights, otherwise weights will be initialized with zero
        vectors and the `trainable` parameter will be set to True.

        Parameters
        ----------
        input_length : int
            The value to set in the `input_length` parameter of the
            `Embedding` layer.

        output_dim : int or None, default None
            In case of creating an Embedding layer with trainable weights
            (we haven't call the load method), this parameters should be set
            to the desired embedding dimensions. If there are loaded vectors,
            this parameters is ignored.

        kwargs
            Additional keyword arguments to pass to the Keras `Embedding` layer.

        Returns
        -------
        Embedding
            A Keras `Embedding` layer.
        """
        if self.vector_len == 0:
            layer = Embedding(
                input_dim=len(self.vocab),
                output_dim=output_dim,
                input_length=input_length,
                trainable=True,
                **kwargs)
        else:
            layer = Embedding(
                input_dim=len(self.vocab),
                output_dim=self.vector_len,
                input_length=input_length,
                weights=[self.vectors],
                **kwargs)
        return layer

    def __getitem__(self, item):
        word_id = self.vocab[item]
        return self.vectors[word_id]

    def __getstate__(self):
        # `logger` object is not pickable. So we remove it.
        state = self.__dict__.copy()
        if 'logger' in state:
            del state['logger']
        return state

    def __setstate__(self, state):
        # Unpickling the object, we don't have a logger instance which is
        # used in some methods. So we create a new one.
        self.__dict__ = state
        self.__dict__['logger'] = logging.getLogger(self.__class__.__name__)


class Glove(WordVectors):
    """
    Load word vectors from a Glove file.
    """

    def __init__(self, vocab, oov_token=None):
        super().__init__(vocab, oov_token)

    def load(self, file_path):
        """
        Load word vectors from a Glove file.
        
        Parameters
        ----------
        file_path : str
            The path to the file.
        """
        line_no = 1
        with open(file_path, 'r') as f:
            self.logger.info(f'Loading word vectors from file "{file_path}".')
            # Read the first line to find the vectors length.
            line = f.readline()
            values = line.split()
            word = values[0]
            self.vector_len = len(values[1:])
            # Create the vocab's vector array.
            oov_id = self._init_vectors()
            # Initialize progress bar.
            progbar = Progbar(len(self.vocab) - 1)  # -PAD
            found = 1
            # Proceed with the rest file.
            while True:
                if word in self.vocab:
                    word_id = self.vocab[word]
                    vector = np.asarray(values[1:], dtype='float32')
                    self.vectors[word_id] = vector
                    progbar.update(found)
                    found += 1
                    if found == len(self.vocab):
                        progbar.update(len(self.vocab) - 1)
                        break
                try:
                    line = f.readline()
                    line_no += 1
                    values = line.split()
                    word = values[0]
                except (EOFError, IndexError):
                    progbar.update(len(self.vocab) - 1)
                    break

            if oov_id is not None:
                self._replace_oov_tokens(oov_id)


class W2V(WordVectors):
    """ Load word vectors from a Word2Vec file. """

    def __init__(self, vocab, oov_token=None):
        super().__init__(vocab, oov_token)

    def load(self, file_path, binary=False):
        """
        Load word vectors from a binary or text word2vec file.

        Parameters
        ----------
        file_path : str
            The path to the file.

        binary : bool, default False
            If the file is text set default to False, otherwise to True.

        Notes
        -----
        The code to load binary files is based on the code found here:
        https://gist.github.com/ottokart/673d82402ad44e69df85
        """
        if binary:
            return self._load_binary_word2vec(file_path)
        else:
            return self._load_text_word2vec(file_path)

    def _load_binary_word2vec(self, file_path):
        """ Load word vectors from a binary word2vec file. """
        float_size = 4  # 32bit float
        with open(file_path, 'rb') as f:
            c = None
            # read the header
            header = ""
            while c != "\n":
                c = f.read(1).decode('utf-8')
                header += c

            num_vectors, self.vector_len = (int(x) for x in header.split())
            self.logger.info(f'Loading {num_vectors} word vectors from file '
                             f'"{file_path}".')
            oov_id = self._init_vectors()
            progbar = Progbar(len(self.vocab) - 1)
            found = 1
            for i in range(num_vectors):
                word = ""
                while True:
                    c = f.read(1).decode('utf-8')
                    if c == " ":
                        break
                    word += c
                word = word.strip()
                binary_vector = f.read(float_size * self.vector_len)
                if word in self.vocab:
                    word_vector = np.asarray([
                        struct.unpack_from('f', binary_vector, i)[0]
                        for i in range(0, len(binary_vector), float_size)
                    ])
                    word_id = self.vocab[word]
                    self.vectors[word_id] = word_vector
                    found += 1
                progbar.update(found)
            if found < len(self.vocab):
                progbar.update(len(self.vocab) - 1)

            if oov_id is not None:
                self._replace_oov_tokens(oov_id)

    def _load_text_word2vec(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            num_vectors, self.vector_len = (int(x) for x in line.split())
            self.logger.info(f'Loading {num_vectors} word vectors from file '
                             f'"{file_path}".')
            oov_id = self._init_vectors()
            found = 1
            progbar = Progbar(len(self.vocab) - 1)
            for i in range(num_vectors):
                line = f.readline()
                values = line.split()
                word = values[0]
                word_vector = np.asarray(values[1:], dtype='float32')
                if word in self.vocab:
                    word_id = self.vocab[word]
                    self.vectors[word_id] = word_vector
                    found += 1
                progbar.update(found)
            if found < len(self.vocab):
                progbar.update(len(self.vocab) - 1)

            if oov_id is not None:
                self._replace_oov_tokens(oov_id)


if __name__ == '__main__':
    import doctest

    doctest.testmod()

