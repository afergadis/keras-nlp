import logging
import abc
import re
from abc import ABC

import numpy as np
from collections import Counter
from keras_preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import Progbar

logger = logging.getLogger(__file__)


def calc_stats(array):
    """
    Calculate basic statistics for a list of numbers.

    Parameters
    ----------
    array : list, 1d array-like
        A list of numbers.

    Returns
    -------
    dict
        The dictionary has keys: min, max, median, mean, std, 25-percentile,
        50-percentile, 75-percentile.

    """
    array = np.asarray(array)
    stats = {
        'min': np.amin(array),
        'max': np.amax(array),
        'median': np.median(array),
        'mean': np.mean(array),
        'std': np.std(array),
        '25-percentile': np.percentile(array, 25),
        '50-percentile': np.percentile(array, 50),
        '75-percentile': np.percentile(array, 75)
    }
    return stats


class Vectorizer:
    """
    An abstract base class for different types of Vectorizers.
    """

    def __init__(self,
                 num_tokens=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 oov_token=None,
                 verbose=1):
        """
        Parameters
        ----------
        num_tokens : int or None, default None
            The maximum number of tokens in the vocabulary of the vectorizer.
            If is set to None, then all the tokens found using `fit_on_texts`
            method will be in the vocabulary.
        filters : str
            Defines the characters that will be removed (replaced with a space
            character) from the texts.
        lower : bool, default True
            Define it the text will be lower cased before processing.
        oov_token : str, None, default None
            If set to a string, then the vocabulary will have an entry that
            represents an out of vocabulary token.
        verbose : int in [0, 2], default 1
            The verbosity of the output during the vectorization methods.
        """

        self.num_tokens = num_tokens
        if filters is not None:
            self.filters = '[' + re.escape(filters) + ']'
        else:
            self.filters = None
        self.lower = lower
        self.oov_token = oov_token

        self.token2id = None
        self.id2token = None
        self.token_counts = Counter()

        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

    def _apply_filters(self, text):
        """ Lowers and removes filters from the text. """
        if self.lower:
            text = text.lower()

        if self.filters is not None:
            text = re.sub(self.filters, ' ', text)

        return text

    def _bag_of_tokens(self):
        """
        Create the `self.token2id` and `self.id2token` vocabularies.
        """
        if self.token2id is None:
            raise ValueError('Please use the `fit_on_texts` method first.')
        if self.num_tokens is not None:
            tokens = self.token_counts.most_common(self.num_tokens)
        else:
            tokens = self.token_counts.most_common()

        if len(tokens) > 0:
            # Tokens are pairs of (token, count). We keep only the token.
            tokens = [token[0] for token in tokens]
            for token in tokens:
                self.token2id[token] = len(self.token2id)
            self.id2token = {v: k for k, v in self.token2id.items()}
        self.num_tokens = len(self.token2id)

    @abc.abstractmethod
    def fit_on_texts(self, texts):
        """
        Creates a tokens vocabulary from the list of texts.

        Parameters
        ----------
        texts : list or list[list]
            Each list item can be a text representing a document or a list of
            texts.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _pad_vectors(self,
                     vectors,
                     shape,
                     padding='pre',
                     truncating='pre',
                     pad_value=0):
        """
        Pad vectors to 2D or 3D arrays.

        A `CharVectorizer` will output list[list] vectors where the outer list
        represent a text and the inner list the tokens. The method will
        ensure that all texts have the same number of tokens and all tokens
        have the same number of characters.
        A `SentWordVectorizer` will output list[list] vectors where the outer
        list represent a sentence and the inner list the tokens. The method will
        ensure that all texts have the same number of sentences and all
        sentences have the same number of tokens.
        A `SentCharVectorizer` will output list[list[list]] vectors. The
        outer list represents the number of sentences of a text, the 2nd
        represents the number of tokens and the 3d the number of characters
        per token.

        Parameters
        ----------
        vectors : list
            A list[list] or list[list[list]] of ids depending on the instance
            of the Vectorizer.

        shape : int, list or tuple
            The target shape of the vectors padded or truncated depending
            on the `shape`'s 1st dimension and the length of `vectors`.

        padding : str, options {'pre', 'post'}, default 'pre'
            Add `pad_value`s to vectors, based on the selected method, in
            order to match the target `shape`.

        truncating : str, options {'pre', 'post'}, default 'pre'
            Delete values, based on the selected method, from `vectors` in
            order to match the target `shape`.

        pad_value : int, default 0
            In case of padding, the value to use.

        Returns
        -------
        ndarray
            An array of shape `shape`.
        """
        raise NotImplementedError()

    @staticmethod
    def _reshape(vectors,
                 num_rows,
                 padding='pre',
                 truncating='pre',
                 pad_value=0):
        """
        Reshape `vectors` to `(num_rows, vectors.shape[1:])`.

        This function is used in order to cut or add rows to the `vectors`
        array, keep the rest dimensions.  This is usefull to make all documents
        of a collection to have the same number of (words, characters) or
        (sentences, words) in case of 2D `vectors`, or (sentences, words,
        characters) in case of 3D `vectors`.

        Parameters
        ----------
        vectors : ndarray
            A 2D or 3D array, depending on the calling instance of `Vectorizer`

        num_rows : int
            The new number or rows of the `vectors`, padded or truncated
            if needed.

        padding: str, options {'pre', 'post'}, default 'pre'
            The padding method to use in case `num_rows > sequences.shape[1]`

        truncating : str, options {'pre', 'post'}, default 'pre'
            The truncating method to use in case `num_rows < sequences.shape[1]`

        pad_value: int, default 0
            The value to use in case of padding.

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            `padding` or `truncating` values are not recognized.
        """
        target_shape = [num_rows] + list(vectors.shape[1:])
        array = np.full(target_shape, fill_value=pad_value)
        if vectors.shape[0] < target_shape[0]:
            if padding == 'pre':
                array[-vectors.shape[0]:, :] = vectors
            elif padding == 'post':
                array[:vectors.shape[0], :] = vectors
            else:
                raise ValueError(
                    'Unknown option `{}` for padding.'.format(padding))
        elif vectors.shape[0] > target_shape[0]:
            if truncating == 'pre':
                array = vectors[-target_shape[0]:, :]
            elif truncating == 'post':
                array = vectors[:target_shape[0], :]
            else:
                raise ValueError(
                    'Unknown option `{}` for truncating.'.format(truncating))
        else:
            array = vectors
        return array

    def _tokens_to_chars(self, tokens):
        """
        For each token create a list of it's character ids.

        Parameters
        ----------
        tokens : list
            The list of tokens to get the character ids from.

        Yields
        ------
        list[list]
            Each list item corresponds to a token. Each token is a list of its
            character ids.
        """
        for token in tokens:
            _chars = []
            for char in token:
                if char in self.token2id:
                    _chars.append(self.token2id[char])
                else:
                    if self.oov_token is not None:
                        _chars.append(self.token2id[self.oov_token])
            yield _chars

    @abc.abstractmethod
    def texts_to_vectors(self,
                         texts,
                         shape=None,
                         padding='pre',
                         truncating='post',
                         pad_value=0):
        """
        Convert a list of texts to a 2D, 3D or 4D array.

        * A `WordVectorizer` will return a 2D array of shape `(len(texts),
        max_tokens)`.
        * A `CharVectorizer` will return a 3D array of shape `(len(texts),
        max_tokens, max_characters)`.
        * A `SentWordVectorizer` will return a 3D array of shape `(len(texts),
        max_sentences, max_tokens)`.
        * A `SentCharVectorizer` will return a 4D array of shape `(len(texts),
        max_sentences, max_tokens, max_characters)`.

        Parameters
        ----------
        texts : list
            The list of texts to convert. Each item represents a document.

        shape : tuple, list or None, default None
            * In case of `WordVectorizer` the shape defines the tuple
            `(max_tokens, )` per text.
            * In case of `CharVectorizer` the shape defines the tuple `(
            max_tokens, max_characters)`.
            * In case of `SentWordVectorizer` the shape defines the tuple `(
            max_sentences, max_tokens)`.
            * In case of `SentCharVectorizer` the shape defines the tuple `(
            max_sentences, max_tokens, max_characters)`.

        padding : str, options {'pre', 'post'}, default 'pre'
            Defines the padding method when the length of tokens/characters
            is less than `max_tokens`/`max_characters`. The filling value is
            the value of the `pad_value` parameter.

        truncating : str, options {'pre', 'post'}, default 'pre'
            Defines the cutting method when  the length of tokens/characters
            is larger than `max_tokens/characters`.

        pad_value : int, default 0
            The value to use when padding is needed.

        Returns
        -------
        ndarray : Numpy 2D array with shape `shape`

        Raises
        ------
        ValueError : `shape` len is not 2 and `shape` is not None.
        """
        raise NotImplementedError()

    def vectors_to_texts(self, vectors):
        """
        Decode `vectors` array to texts using the Vectorizer's vocabulary.

        Parameters
        ----------
        vectors : ndarray
            The shape of `vectors` may be:
            * 2D `(num_of_texts, num_of_tokens)` from `WordVectorizer`;
            * 3D `(num_of_texts, num_of_sentences, num_of_words)` from
            `SentWordVectorizer`;
            * 3D `(num_of_texts, num_of_words, num_of_characters)` from
            `CharVectorizer`;
            a 4D `(num_of_texts, num_of_sentences, num_of_words,
            num_of_characters)` from `SentCharVectorizer`.

        Returns
        -------
        list
            The list has length of `num_of_texts`.
            * In case of 2D input, the list has the words of the texts.
            * In case of 3D input, the list has a list sentences. Each item of
            the nested list is a word in case of `SentWordVectorizer`,
            or the list has a list of words. Each item of the word's list is
            the characters of the words.
            * In case of 4D input, the output is a list of texts with their list
            of sentences. For each sentence a list of tokens where each token
            is a list of its characters.
        """

        def decode_vector(vector):
            text = []
            for values in vector:
                tokens = [
                    self.id2token[t] for t in values
                    if t != self.token2id['_PAD_']
                ]
                if len(tokens) > 0:
                    text.append(tokens)
            return text

        if self.id2token is None:
            raise ValueError('Please use `fit_on_texts` method first.')
        self.logger.info('Converting vectors to texts.')
        texts = []
        progbar = Progbar(
            vectors.shape[0], interval=0.25, verbose=self.verbose)
        if vectors.ndim == 2:
            for doc in vectors:
                # Just to be able to update progress bar.
                text = decode_vector(doc.reshape(1, -1))
                # text is a list with one item; the words of the document.
                texts.append(text[0])
                progbar.update(len(texts))
        elif vectors.ndim == 3:
            for vector in vectors:
                decoded = decode_vector(vector)
                if len(decoded) > 0:  # If len==0, it is padded.
                    texts.append(decoded)
                progbar.update(len(texts))
        elif vectors.ndim == 4:
            for doc in vectors:
                text = []
                for sents in doc:
                    decoded = decode_vector(sents)
                    if len(decoded) > 0:  # If len==0, it is padded.
                        text.append(decoded)
                texts.append(text)
                progbar.update(len(texts))

        return texts

    @abc.abstractmethod
    def stats(self):
        """
        Get statistics about the texts the Vectorizer is fitted on.

        Returns
        -------
        dict
        """
        raise NotImplementedError()

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


class CharVectorizer(Vectorizer):
    """
    Convert a list of texts to an array of vectors with shape `(num_of_texts,
    max_words, max_characters)`.

    For each text (document) keep a maximum number of words. Each word is
    represented with a list of `max_characters` ids.

    Examples
    --------
    >>> char_vectorizer = CharVectorizer(oov_token='?', \
    characters='abcdefghijklmnopqrstuvwxyz', verbose=0)
    >>> texts = ['Phasellus fermentum tellus eget libero sodales varius.', \
    'In vestibulum erat nec nulla porttitor dignissim.']
    >>> # In case `characters` are set, fit_on_texts it's secondary.
    >>> char_vectorizer.fit_on_texts(texts)  #doctest:+ELLIPSIS
    ...
    >>> docs = ['Nam accumsan velit vel ligula convallis cursus.', \
    'Nulla porttitor felis risus, vitae facilisis massa consectetur id.']
    >>> vectors = char_vectorizer.texts_to_vectors(docs)  #doctest:+ELLIPSIS
    ...
    >>> print(vectors.shape)  # (len(texts), max_words, max_characters)
    (2, 9, 11)
    >>> decoded = char_vectorizer.vectors_to_texts(vectors)
    >>> print(decoded[0][:2])  # First 2 words of the 1st doc in docs.
    [['n', 'a', 'm'], ['a', 'c', 'c', 'u', 'm', 's', 'a', 'n']]
    """

    def __init__(self,
                 word_tokenize=None,
                 characters=None,
                 num_chars=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 oov_token=None,
                 verbose=1):
        """
        Parameters
        ----------
        word_tokenize : callable or None, default None
            If set, then the function should return a list of tokens (words)
            for a given text. When `word_tokenize` is set to None,
            the `split()` method is used for tokenization.

        characters : str, list or None, default None
            If set then only the characters defined will be in the vectrorizer's
            vocabulary. Otherwise, the characters found in the texts during
            `fit_on_texts` method will be used.

        num_chars : int or None, default None
            The maximum number of characters in the vocabulary of the
            vectorizer. If is set to None, then all the characters found using
            `fit_on_texts` method will be in the vocabulary.

        filters : str
            Defines the characters that will be removed (replaced with a space
            character) from the texts.

        lower : bool, default True
            Define if the text will be lower cased before processing.

        oov_token : str, None, default None
            If set to a string, then the vocabulary will have an entry that
            represents an out of vocabulary token.

        verbose : int in [0, 2], default 1
            The verbosity of the output during the vectorization methods.

        Attributes
        ----------
        num_tokens : int
            The number of tokens (characters) in the vocabulary.
        num_chars : int
            Alias for the `num_tokens` attribute.
        words_stats : dict
            A dictionary with statistics about the tokens (words) of the texts
            (min/max/std/mean/median/percentiles of words among texts).
        chars_stats : dict
            A dictionary with statistics about the characters of the words.
            (min/max/std/mean/median/percentiles of characters among words).
        """
        super(CharVectorizer, self).__init__(num_chars, filters, lower,
                                             oov_token, verbose)
        self.word_tokenize = word_tokenize
        self.num_chars = num_chars

        if characters is None:
            self.characters = None
        elif isinstance(characters, str):
            self.characters = [c for c in characters]
            self.fit_on_texts([])
        else:
            self.characters = characters
            self.fit_on_texts([])

        self.words_stats = None
        self.chars_stats = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def fit_on_texts(self, texts):
        self.logger.info('Creating vocabulary.')
        self.token2id = {'_PAD_': 0}
        if self.oov_token is not None:
            self.token2id[self.oov_token] = 1
        if self.characters is None:
            progbar = Progbar(len(texts), interval=0.25, verbose=self.verbose)
            for i, text in enumerate(texts):
                if isinstance(text, list):
                    text = self._apply_filters(' '.join(text))
                else:
                    text = self._apply_filters(text)
                chars = [c for c in text if c != ' ']
                self.token_counts.update(chars)
                progbar.update(i)
            progbar.update(len(texts))  # Finalize
        else:
            self.token_counts.update(self.characters)
        self._bag_of_tokens()
        self.num_chars = self.num_tokens

    def _pad_vectors(self,
                     sequences,
                     shape,
                     padding='pre',
                     truncating='pre',
                     pad_value=0):
        if len(shape) != 2:
            raise ValueError(f'The `shape` should be of rank 2 defining the'
                             f'maximum words per text and the maximum '
                             f'characters per word. Found a `shape` '
                             f'with rank {len(shape)}.')
        max_words, max_characters = shape
        vectors = np.full(
            shape=(len(sequences), max_words, max_characters),
            fill_value=self.token2id['_PAD_'],
            dtype=int)
        self.logger.info(f'Reshaping vectors to shape {shape}.')
        progbar = Progbar(len(sequences), interval=0.25, verbose=self.verbose)
        for i, doc in enumerate(sequences):
            chars_vector = pad_sequences(
                doc,
                max_characters,
                padding=padding,
                truncating=truncating,
                value=pad_value)
            vectors[i] = self._reshape(chars_vector, max_words, padding,
                                       truncating, pad_value)
            progbar.update(i)
        progbar.update(len(sequences))  # Finalize
        return vectors

    def texts_to_vectors(self,
                         texts: list,
                         shape=None,
                         padding='pre',
                         truncating='pre',
                         pad_value=0):
        if shape is not None:
            if len(shape) != 2:
                raise ValueError(
                    f'The `shape` should be of rank 2 defining the'
                    f'maximum words per text and the maximum '
                    f'characters per word. Found a shape '
                    f'with rank {len(shape)}.')
        _texts = []
        self.logger.info('Converting texts to vectors.')
        progbar = Progbar(len(texts), interval=0.25, verbose=self.verbose)
        for text in texts:
            text = self._apply_filters(text)
            if self.word_tokenize is None:
                words = text.split()
            else:
                words = self.word_tokenize(text)
            # In some rare cases, all the characters of a word have been
            # removed because of the filter characters. So the length of the
            # word is zero.
            if len(words) == 0:
                _words = [[0]]  # The list of characters in a word.
            else:
                _words = list(self._tokens_to_chars(words))
            _texts.append(_words)
            progbar.update(len(_texts))

        # Calculate document length in words.
        texts_len = [len(text) for text in _texts]
        words_len = [len(word) for doc in _texts for word in doc]

        self.words_stats = calc_stats(texts_len)
        self.chars_stats = calc_stats(words_len)

        if shape is None:
            max_words = self.words_stats['max']
            max_characters = self.chars_stats['max']
            shape = (max_words, max_characters)
        vectors = self._pad_vectors(_texts, shape, padding, truncating,
                                    pad_value)
        return vectors

    def stats(self):
        return self.words_stats, self.chars_stats

    def __str__(self):
        if self.token2id is None:
            msg = 'CharVectorizer(Vocab Size: 0)'
        else:
            msg = 'CharVectorizer(Vocab Size: {})'.format(len(self.token2id))
        return msg

    def __repr__(self):
        return self.__str__()


class WordVectorizer(Vectorizer):
    """
    Convert a list of texts to an array of vectors with shape `(num_of_texts,
    max_words)`.

    Examples
    --------
    >>> word_vectorizer = WordVectorizer(verbose=0)
    >>> texts = ['Phasellus fermentum tellus eget libero sodales varius.', \
    'In vestibulum erat nec nulla porttitor dignissim.']
    >>> word_vectorizer.fit_on_texts(texts)  #doctest:+ELLIPSIS
    ...
    >>> vectors = word_vectorizer.texts_to_vectors(texts)  #doctest:+ELLIPSIS
    ...
    >>> print(vectors.shape)  # (len(texts), max_words)
    (2, 7)
    >>> decoded = word_vectorizer.vectors_to_texts(vectors)
    >>> print(decoded[0][:3])  # First 3 words of the 1st doc in docs.
    ['phasellus', 'fermentum', 'tellus']
    """

    def __init__(self,
                 word_tokenize=None,
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 oov_token=None,
                 verbose=1):
        """
        Parameters
        ----------
        word_tokenize : callable or None, default None
            If set, then the function should return a list of tokens (words)
            for a given text. When `word_tokenize` is set to None,
            the `split()` method is used for tokenization.

        num_words : int or None, default None
            The maximum number of words in the vocabulary of the
            vectorizer. If is set to None, then all the words found using
            `fit_on_texts` method will be in the vocabulary.

        filters : str
            Defines the characters that will be removed (replaced with a space
            character) from the texts.

        lower : bool, default True
            Define if the text will be lower cased before processing.

        oov_token : str, None, default None
            If set to a string, then the vocabulary will have an entry that
            represents an out of vocabulary token.

        verbose : int in [0, 2], default 1
            The verbosity of the output during the vectorization methods.

        Attributes
        ----------
        num_tokens : int
            The number of tokens (words) in the vocabulary.

        num_words : int
            Alias for the `num_tokens` attribute.

        words_stats : dict
            A dictionary with statistics about the tokens (words) of the texts
            (min/max/std/mean/median/percentiles of words among texts).
        """
        super(WordVectorizer, self).__init__(num_words, filters, lower,
                                             oov_token, verbose)
        self.word_tokenize = word_tokenize
        self.words_stats = []

        self.logger = logging.getLogger(self.__class__.__name__)

    def fit_on_texts(self, texts):
        self.token2id = {'_PAD_': 0}
        if self.oov_token is not None:
            self.token2id[self.oov_token] = 1

        self.logger.info('Creating vocabulary.')
        progbar = Progbar(len(texts), interval=0.25, verbose=self.verbose)
        for i, text in enumerate(texts):
            if isinstance(text, list):
                text = self._apply_filters(' '.join(text))
            else:
                text = self._apply_filters(text)
            if self.word_tokenize is not None:
                tokens = self.word_tokenize(text)
            else:
                tokens = text.split()
            self.token_counts.update(tokens)
            progbar.update(i)
        progbar.update(len(texts))
        self._bag_of_tokens()

    def _pad_vectors(self,
                     vectors,
                     shape,
                     padding='pre',
                     truncating='pre',
                     pad_value=0):
        pass

    def texts_to_vectors(self,
                         texts: list,
                         shape=None,
                         padding='pre',
                         truncating='pre',
                         pad_value=0):
        if shape is not None:
            if len(shape) != 1:
                raise ValueError(
                    f'The `shape` should be of rank 1 defining the'
                    f'maximum words per text. Found a shape '
                    f'with rank {len(shape)}.')
        _texts = []
        self.logger.info('Converting texts to vectors.')
        progbar = Progbar(len(texts), interval=0.25, verbose=self.verbose)
        for text in texts:
            _words = []
            text = self._apply_filters(text)
            if self.word_tokenize is None:
                words = text.split()
            else:
                words = self.word_tokenize(text)
            for word in words:
                if word in self.token2id:
                    _words.append(self.token2id[word])
                else:
                    if self.oov_token is not None:
                        _words.append(self.token2id[self.oov_token])
            _texts.append(_words)
            progbar.update(len(_texts))

        docs_len = [len(doc) for doc in _texts]
        self.words_stats = calc_stats(docs_len)

        if shape is None:
            shape = (self.words_stats['max'], )

        vectors = pad_sequences(
            _texts,
            shape[0],
            padding=padding,
            truncating=truncating,
            value=self.token2id['_PAD_'])
        return vectors

    def stats(self):
        return self.words_stats

    def __str__(self):
        if self.token2id is None:
            msg = 'WordVectorizer(Vocab Size: 0)'
        else:
            msg = '<WordVectorizer(Vocab Size: {})'.format(len(self.token2id))
        return msg

    def __repr__(self):
        return self.__str__()


class SentCharVectorizer(CharVectorizer):
    """
    Convert a list of texts to an array of vectors with shape `(num_of_texts,
    max_sentences, max_words, max_characters)`.

    Examples
    --------
    >>> sent_char_vectorizer = SentCharVectorizer(verbose=0)
    >>> # Two documents. The fists with two sentences and the second with one.
    >>> # The 1st document is already tokenized on sentences. Alternately, you
    >>> # may pass a sent_tokenizer callable.
    >>> texts = [['Phasellus fermentum tellus eget libero sodales varius.', \
    'In vestibulum erat nec nulla porttitor dignissim.'], \
    ['Nam accumsan velit vel ligula convallis cursus.'] \
    ]
    >>> sent_char_vectorizer.fit_on_texts(texts)  #doctest:+ELLIPSIS
    ...
    >>> vectors = sent_char_vectorizer.texts_to_vectors(texts) #doctest:+ELLIPSIS
    ...
    >>> print(vectors.shape)  # (len(texts), max_sentences, max_words, max_characters)
    (2, 2, 7, 10)
    >>> decoded = sent_char_vectorizer.vectors_to_texts(vectors)
    >>> print(decoded[0][1][:2])  # 1st text, 2d sentence, 2 words
    [['i', 'n'], ['v', 'e', 's', 't', 'i', 'b', 'u', 'l', 'u', 'm']]
    """

    def __init__(self,
                 sent_tokenize=None,
                 word_tokenize=None,
                 characters=None,
                 num_chars=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 oov_token=None,
                 verbose=1):
        """
        Parameters
        ----------
        sent_tokenize : callable or None, default None
            If set, then the function should return a list of sentences
            for a given text. When `sent_tokenize` is set to None, we assume
            that the texts are already tokenized. So each text is expected to
            be a list of sentences.

        word_tokenize : callable or None, default None
            If set, then the function should return a list of tokens (words)
            for a given text. When `word_tokenize` is set to None,
            the `split()` method is used for tokenization.

        characters : str, list or None, default None
            If set then only the characters defined will be in the vectrorizer's
            vocabulary. Otherwise, the characters found in the texts during
            `fit_on_texts` method will be used.

        num_chars : int or None, default None
            The maximum number of characters in the vocabulary of the
            vectorizer. If is set to None, then all the characters found using
            `fit_on_texts` method will be in the vocabulary.

        filters : str
            Defines the characters that will be removed (replaced with a space
            character) from the texts.

        lower : bool, default True
            Define if the text will be lower cased before processing.

        oov_token : str, None, default None
            If set to a string, then the vocabulary will have an entry that
            represents an out of vocabulary token.

        verbose : int in [0, 2], default 1
            The verbosity of the output during the vectorization methods.

        Attributes
        ----------
        num_tokens : int
            The number of tokens (characters) in the vocabulary.

        num_words : int
            Alias for the `num_tokens` attribute.

        sents_stats : dict
            A dictionary with statistics about the sentences of the texts.
            (min/max/std/mean/median/percentiles of sentences among texts).

        words_stats : dict
            A dictionary with statistics about the tokens (words) of the texts
            (min/max/std/mean/median/percentiles of words among texts).

        chars_stats : dict
            A dictionary with statistics about the characters of the words.
            (min/max/std/mean/median/percentiles of characters among words).
        """
        super(SentCharVectorizer,
              self).__init__(word_tokenize, characters, num_chars, filters,
                             lower, oov_token, verbose)
        self.sent_tokenize = sent_tokenize
        self.sents_stats = None
        self.words_stats = None
        self.chars_stats = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def _pad_vectors(self,
                     sequences,
                     shape,
                     padding='pre',
                     truncating='pre',
                     pad_value=0):
        if len(shape) != 3:
            raise ValueError('`shape` should be a tuple with three values.')
        # doc, sent, word, character
        max_sentences, max_words, max_characters = shape
        vectors = np.full(
            shape=(len(sequences), max_sentences, max_words, max_characters),
            fill_value=self.token2id['_PAD_'],
            dtype=int)

        self.logger.info(f'Reshaping vectors to shape {shape}.')
        progbar = Progbar(len(sequences), interval=0.25, verbose=self.verbose)
        for i in range(len(sequences)):
            num_sentences = len(sequences[i])
            words_chars_vector = np.full(
                shape=(num_sentences, max_words, max_characters),
                fill_value=self.token2id['_PAD_'],
                dtype=int)
            for j in range(num_sentences):
                chars_vector = pad_sequences(
                    sequences[i][j],
                    max_characters,
                    padding=padding,
                    truncating=truncating,
                    value=pad_value)
                words_chars_vector[j] = self._reshape(
                    chars_vector, max_words, padding, truncating, pad_value)
            # Put the words_chars_vector into the document vector.
            vectors[i] = self._reshape(words_chars_vector, max_sentences,
                                       padding, truncating, pad_value)
            progbar.update(i)
        progbar.update(len(sequences))  # Finalize

        return vectors

    def texts_to_vectors(self,
                         texts,
                         shape=None,
                         padding='pre',
                         truncating='pre',
                         pad_value=0):
        if shape is not None:
            if len(shape) != 3:
                raise ValueError(
                    f'The `shape` should be of rank 3 defining the'
                    f'maximum number of sentences per text, '
                    f'maximum words per sentence and the maximum '
                    f'characters per word. Found a shape '
                    f'with rank {len(shape)}.')

        _texts = []
        self.logger.info('Converting texts to vectors.')
        progbar = Progbar(len(texts), interval=0.25, verbose=self.verbose)
        for text in texts:
            _text = []
            if self.sent_tokenize is None:
                # We assume that `texts` is a document already tokenized.
                # So, each `text` is a "list" of one sentence.
                if not isinstance(text, list):
                    raise ValueError(
                        'For a sentence tokenized list of texts, '
                        'each text should be a list of sentences.')
                sentences = text
            else:
                sentences = self.sent_tokenize(text)
            for sentence in sentences:
                sent = self._apply_filters(sentence)
                if self.word_tokenize is None:
                    words = sent.split()
                else:
                    words = self.word_tokenize(sent)
                if len(words) == 0:
                    _words = [[0]]  # The list of characters in a word.
                else:
                    _words = list(self._tokens_to_chars(words))
                _text.append(_words)
            _texts.append(_text)
            progbar.update(len(_texts))

        # Calculate document length in sentences, sentence length in words
        # and words len in characters.
        docs_len = [len(sents) for sents in _texts]
        sents_len = [len(words) for sents in _texts for words in sents]
        words_len = [
            len(chars) for sents in _texts for words in sents
            for chars in words
        ]

        self.sents_stats = calc_stats(docs_len)
        self.words_stats = calc_stats(sents_len)
        self.chars_stats = calc_stats(words_len)

        if shape is None:
            max_sentences = self.sents_stats['max']
            max_words = self.words_stats['max']
            max_characters = self.chars_stats['max']
            shape = (max_sentences, max_words, max_characters)
        vectors = self._pad_vectors(_texts, shape, padding, truncating,
                                    pad_value)
        return vectors

    def stats(self):
        return self.sents_stats, self.words_stats, self.chars_stats

    def __str__(self):
        if self.token2id is None:
            msg = 'SentCharVectorizer(Vocab Size: 0)'
        else:
            msg = 'SentCharVectorizer(Vocab size: {})'.format(
                len(self.token2id))
        return msg

    def __repr__(self):
        return self.__str__()


class SentWordVectorizer(WordVectorizer):
    """
    Convert a list of texts to an array with shape `(num_of_texts,
    max_sentences, max_words)`.

    Examples
    --------
    >>> sent_word_vectorizer = SentWordVectorizer(verbose=0)
    >>> # Two documents. The fists with two sentences and the second with one.
    >>> # The 1st document is already tokenized on sentences. Alternately, you
    >>> # may pass a sent_tokenizer callable.
    >>> texts = [['Phasellus fermentum tellus sodales varius.', \
    'In vestibulum erat nec nulla porttitor dignissim.'], \
    ['Nam accumsan velit vel ligula convallis.'] \
    ]
    >>> sent_word_vectorizer.fit_on_texts(texts)  #doctest:+ELLIPSIS
    ...
    >>> vectors = sent_word_vectorizer.texts_to_vectors(texts) #doctest:+ELLIPSIS
    ...
    >>> print(vectors.shape)  # (len(texts), max_sentences, max_words)
    (2, 2, 7)
    >>> decoded = sent_word_vectorizer.vectors_to_texts(vectors)
    >>> print(decoded[0][1][:2])  # 1st text, 2d sentence, 2 words
    ['in', 'vestibulum']
    """

    def __init__(self,
                 sent_tokenize=None,
                 word_tokenize=None,
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 oov_token=None,
                 verbose=1):
        """
        Parameters
        ----------
        sent_tokenize : callable or None, default None
            If set, then the function should return a list of sentences
            for a given text. When `sent_tokenize` is set to None, we assume
            that the texts are already tokenized. So each text is expected to
            be a list of sentences.

        word_tokenize : callable or None, default None
            If set, then the function should return a list of tokens (words)
            for a given text. When `word_tokenize` is set to None,
            the `split()` method is used for tokenization.

        num_words : int or None, default None
            The maximum number of characters in the vocabulary of the
            vectorizer. If is set to None, then all the characters found using
            `fit_on_texts` method will be in the vocabulary.

        filters : str
            Defines the characters that will be removed (replaced with a space
            character) from the texts.

        lower : bool, default True
            Define if the text will be lower cased before processing.

        oov_token : str, None, default None
            If set to a string, then the vocabulary will have an entry that
            represents an out of vocabulary token.

        verbose : int in [0, 2], default 1
            The verbosity of the output during the vectorization methods.

        Attributes
        ----------
        num_tokens : int
            The number of tokens (characters) in the vocabulary.

        num_words : int
            Alias for the `num_tokens` attribute.

        sents_stats : dict
            A dictionary with statistics about the sentences of the texts.
            (min/max/std/mean/median/percentiles of sentences among texts).

        words_stats : dict
            A dictionary with statistics about the tokens (words) of the texts
            (min/max/std/mean/median/percentiles of words among texts).

        chars_stats : dict
            A dictionary with statistics about the characters of the words.
            (min/max/std/mean/median/percentiles of characters among words).
        """
        super(SentWordVectorizer, self).__init__(
            word_tokenize, num_words, filters, lower, oov_token, verbose)
        self.sent_tokenize = sent_tokenize

        self.sents_stats = []
        self.words_stats = []
        self.chars_stats = []

    def _pad_vectors(self,
                     sequences,
                     shape,
                     padding='pre',
                     truncating='pre',
                     pad_value=0):
        max_sentences, max_words = shape
        vectors = np.full(
            shape=(len(sequences), max_sentences, max_words),
            fill_value=self.token2id['_PAD_'])
        self.logger.info(f'Reshaping vectors to shape {shape}.')
        progbar = Progbar(len(sequences), interval=0.25, verbose=self.verbose)
        for i in range(len(sequences)):
            words_vector = pad_sequences(
                sequences[i],
                max_words,
                padding=padding,
                truncating=truncating,
                value=pad_value)
            vectors[i] = self._reshape(
                words_vector,
                num_rows=max_sentences,
                padding=padding,
                truncating=truncating,
                pad_value=pad_value)
            progbar.update(i)
        progbar.update(len(sequences))  # Finalize

        return vectors

    def texts_to_vectors(self,
                         texts,
                         shape=None,
                         padding='pre',
                         truncating='pre',
                         pad_value=0):
        if shape is not None:
            if len(shape) != 2:
                raise ValueError(
                    f'The `shape` should be of rank 2 defining the'
                    f'maximum sentences per text and the maximum '
                    f'words per sentence. Found a shape '
                    f'with rank {len(shape)}.')
        _texts = []
        self.logger.info('Converting texts to vectors.')
        progbar = Progbar(len(texts), interval=0.25, verbose=self.verbose)
        for text in texts:
            _text = []
            if self.sent_tokenize is None:
                # We assume that `texts` is a document already tokenized.
                # So, each `text` is a "list" of one sentence.
                if not isinstance(text, list):
                    raise ValueError(
                        'For a sentence tokenized list of texts, '
                        'each text should be a list of sentences.')
                sentences = text
            else:
                sentences = self.sent_tokenize(text)
            for sentence in sentences:
                text = self._apply_filters(sentence)
                if self.word_tokenize is None:
                    words = text.split()
                else:
                    words = self.word_tokenize(text)
                _words = []
                for word in words:
                    if word in self.token2id:
                        _words.append(self.token2id[word])
                    else:
                        if self.oov_token is not None:
                            _words.append(self.token2id[self.oov_token])
                _text.append(_words)
            _texts.append(_text)
            progbar.update(len(_texts))

        docs_len = [len(doc) for doc in _texts]
        sents_len = [len(sent) for doc in _texts for sent in doc]
        # Words are numbers. To calculate length we get the real word back
        # using self.id2word
        words_len = [
            len(self.id2token[word]) for doc in _texts for sent in doc
            for word in sent
        ]
        self.sents_stats = calc_stats(docs_len)
        self.words_stats = calc_stats(sents_len)
        self.chars_stats = calc_stats(words_len)

        if shape is None:
            shape = self.sents_stats['max'], self.words_stats['max']

        vectors = self._pad_vectors(
            _texts,
            shape,
            padding=padding,
            truncating=truncating,
            pad_value=self.token2id['_PAD_'])
        return vectors

    def stats(self):
        return self.sents_stats, self.words_stats, self.chars_stats

    def __str__(self):
        if self.token2id is None:
            msg = 'SentWordVectorizer(Vocab Size: 0)'
        else:
            msg = 'SentWordVectorizer(Vocab Size: {})'.format(
                len(self.token2id))
        return msg

    def __repr__(self):
        return self.__str__()
