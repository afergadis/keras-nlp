# Keras NLP
This is a library with useful functionality to prepare text for Keras models.

## The Problem
We have list of texts (documents) and we want to convert them to list of 
ids. The easy case is when we want just a sequence of words per text.  Using
the Keras tools we can do:
```pydocstring
>>> from keras.preprocessing.text import Tokenizer
>>> from keras_preprocessing.sequence import pad_sequences
>>> texts = ['Phasellus fermentum tellus eget libero sodales varius.', \
'In vestibulum erat nec nulla porttitor.']
>>> tokenizer = Tokenizer()
>>> tokenizer.fit_on_texts(texts)
>>> sequences = tokenizer.texts_to_sequences(texts)
>>> vectors = pad_sequences(sequences, maxlen=8)
>>> print(vectors)
[[ 0,  1,  2,  3,  4,  5,  6,  7]
 [ 0,  0,  8,  9, 10, 11, 12, 13]]
```

In case we want the characters we have to do:
```pydocstring
>>> from keras.preprocessing.text import Tokenizer
>>> from keras_preprocessing.sequence import pad_sequences
>>> texts = ['Phasellus fermentum tellus eget libero sodales varius.', \
'In vestibulum erat nec nulla porttitor.']
>>> tokenizer = Tokenizer(char_level=True)
>>> tokenizer.fit_on_texts(texts)
>>> sequences = tokenizer.texts_to_sequences(texts)
>>> vectors = pad_sequences(sequences, maxlen=15)
>>> print(vectors)
[[ 5, 11, 20,  8,  3,  1,  5,  2, 15,  8,  7,  9,  6,  5, 16]
 [ 6,  3,  3,  8,  2, 13, 11,  7,  4,  4,  9,  4, 11,  7, 16]]
```

*But* what if we want to split texts on their sentences first and keep a fix
number of sentences per text and a fix number of words per sentence? 
Then we have to split sentences and pad or truncate to have the same number
among texts. Also, for all texts we have to pad or truncate in order to have
the same number of words per sentence.
And what if we want sentences, words and characters per word, all of them in
fixed numbers?  Of course we can do those writing some more lines of code, 
**or** we can just use *Vectorizers*.

## A new Approach 

The main functionality provided is the *Vectorization* of texts.
Input texts are tokenized and converted to numbers, padded or truncated to
the given shape.

There are four types of Vectorizers.
* WordVectorizer
* CharVectorizer
* SentWordVectorizer
* SentCharVectorizer

### Word Vectorizer
The `WordVectorizer` will take as input a list of texts and will return an 
array of shape `(num_of_texts, max_words)`. The `max_words` is the number of
maximum words per text and is the same for all texts. As a number it can be
given or inferred from the texts.
  
```pydocstring
>>> from keras_nlp import WordVectorizer
>>> word_vectorizer = WordVectorizer(verbose=0)
>>> texts = ['Phasellus fermentum tellus eget libero sodales varius.', \
'In vestibulum erat nec nulla porttitor.']
>>> word_vectorizer.fit_on_texts(texts)  #doctest:+ELLIPSIS
...
>>> vectors = word_vectorizer.texts_to_vectors(texts, shape=(8, ))  #doctest :+ELLIPSIS
...
>>> print(vectors.shape)  # (len(texts), max_words)
(2, 8)
>>> print(vectors)
[[ 0  1  2  3  4  5  6  7]
 [ 0  0  8  9 10 11 12 13]]
>>> decoded = word_vectorizer.vectors_to_texts(vectors)
>>> print(decoded[0][:3])  # First 3 words of the 1st text in texts.
['phasellus', 'fermentum', 'tellus']
```    

### Char Vectorizer
A `CharVectorizer` will tokenize the input to its characters. So the output
for a list of texts will be an array of shape `(num_of_texts, max_words
, max_characters)`. So each word in the text will be tokenized into each
characters and keep `max_characters` for all words among texts. Each
characters is substituted by its id.
   
```pydocstring
>>> from keras_nlp import CharVectorizer
>>> char_vectorizer = CharVectorizer(oov_token='?', \
characters='abcdefghijklmnopqrstuvwxyz', verbose=0)
>>> texts = ['Phasellus fermentum tellus eget libero sodales varius.', \
'In vestibulum erat nec nulla porttitor.']
>>> char_vectorizer.fit_on_texts(texts)  #doctest:+ELLIPSIS
...
>>> docs = ['Nam accumsan velit vel ligula convallis cursus.', \
'Nulla porttitor felis risus, vitae facilisis massa consectetur id.']
>>> vectors = char_vectorizer.texts_to_vectors(docs, shape=(5, 8))  #doctest :+ELLIPSIS
...
>>> print(vectors.shape)  # (len(texts), max_words, max_characters)
(2, 5, 8)
>>> print(vectors)
[[[ 0  0  0 23  6 13 10 21]
  [ 0  0  0  0  0 23  6 13]
  [ 0  0 13 10  8 22 13  2]
  [16 15 23  2 13 13 10 20]
  [ 0  0  4 22 19 20 22 20]]
 [[ 0  0  0 23 10 21  2  6]
  [ 2  4 10 13 10 20 10 20]
  [ 0  0  0 14  2 20 20  2]
  [20  6  4 21  6 21 22 19]
  [ 0  0  0  0  0  0 10  5]]]
>>> decoded = char_vectorizer.vectors_to_texts(vectors)
>>> print(decoded[0][:2])  # First 2 words of the 1st doc in docs.
>>> # *Attention:* Words are truncated! 1st doc looses the first 2 words.
[['v', 'e', 'l', 'i', 't'], ['v', 'e', 'l']]
```    

### Sent Word Vectorizer
This is a `WordVectorizer` that also splits the input texts to sentences. The
resulting array is of shape `(num_of_texts, max_sentences, max_words)`. For
all texts we have the same number of sentences `max_sentences` and the
same number of words per sentence `max_words`.
   
```pydocstring
>>> from keras_nlp import SentWordVectorizer
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
>>> print(vectors)
[[[ 0  0  1  2  3  4  5]
  [ 6  7  8  9 10 11 12]]
 [[ 0  0  0  0  0  0  0]
  [ 0 13 14 15 16 17 18]]]
>>> decoded = sent_word_vectorizer.vectors_to_texts(vectors)
>>> print(decoded[0][1][:2])  # 1st text, 2d sentence, 2 words
['in', 'vestibulum']
```

 The same example using `sent_tokenize`.
```pydocstring
>>> from keras_nlp import SentWordVectorizer
>>> from keras_nlp.preprocessing import sent_tokenize
>>> sent_word_vectorizer = SentWordVectorizer(sent_tokenize, verbose=0)
>>> # Two documents. The fists with two sentences and the second with one.
>>> texts = ['Phasellus fermentum tellus sodales varius. ' \
'In vestibulum erat nec nulla porttitor dignissim.', \
'Nam accumsan velit vel ligula convallis.' \
]
>>> sent_word_vectorizer.fit_on_texts(texts)  #doctest:+ELLIPSIS
...
>>> vectors = sent_word_vectorizer.texts_to_vectors(texts) #doctest:+ELLIPSIS
...
>>> print(vectors.shape)  # (len(texts), max_sentences, max_words)
(2, 2, 7)
>> print(vectors)
[[[ 0  0  1  2  3  4  5]
  [ 6  7  8  9 10 11 12]]
 [[ 0  0  0  0  0  0  0]
  [ 0 13 14 15 16 17 18]]]
>>> decoded = sent_word_vectorizer.vectors_to_texts(vectors)
>>> print(decoded[0][1][:2])  # 1st text, 2d sentence, 2 words
['in', 'vestibulum']
```

### Sent Char Vectorizer
An extension to the `CharVectorizer` to also split the input texts to
sentences. The resulting array is of shape `(num_of_texts, max_sentences
, max_words, max_characters)`. All texts have the same number of sentences. 
Each sentence has the same number of words, and each word has the same
number of characters.
  
```pydocstring
>>> from keras_nlp import SentCharVectorizer
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
>>> print(vectors[0])  # 2 sentences of the 1st text, 7 words/sent, 10 chars/word.
[[[ 0 15 18  6  2  3  1  1  4  2]
  [ 0 19  3  9 10  3  8  7  4 10]
  [ 0  0  0  0  7  3  1  1  4  2]
  [ 0  0  0  0  0  0  3 14  3  7]
  [ 0  0  0  0  1  5 16  3  9 11]
  [ 0  0  0  2 11 17  6  1  3  2]
  [ 0  0  0  0 12  6  9  5  4  2]]
 [[ 0  0  0  0  0  0  0  0  5  8]
  [12  3  2  7  5 16  4  1  4 10]
  [ 0  0  0  0  0  0  3  9  6  7]
  [ 0  0  0  0  0  0  0  8  3 13]
  [ 0  0  0  0  0  8  4  1  1  6]
  [ 0 15 11  9  7  7  5  7 11  9]
  [ 0 17  5 14  8  5  2  2  5 10]]]
>>> decoded = sent_char_vectorizer.vectors_to_texts(vectors)
>>> print(decoded[0][1][:2])  # 1st text, 2d sentence, 2 words
[['i', 'n'], ['v', 'e', 's', 't', 'i', 'b', 'u', 'l', 'u', 'm']]
```
## Mappings
Two helper classes, `Glove` and `W2V` provide functionality to load word
vectors and return an embedding layer for a given vocabulary.

```pydocstring
>>> import tempfile
>>> from keras_nlp import Glove, WordVectorizer
>>> vectors_file = tempfile.NamedTemporaryFile(encoding='utf-8')
>>> vectors_file.write(b'phasellus 0.1 -0.3 0.2\n')
>>> vectors_file.write(b'fermentum 0.2 0.1 -0.1\n')
>>> vectors_file.seek(0)
>>> word_vectorizer = WordVectorizer(oov_token='_UNK_', verbose=0)
>>> texts = ['Phasellus fermentum tellus eget libero sodales varius.', \
'In vestibulum erat nec nulla porttitor.']
>>> word_vectorizer.fit_on_texts(texts)  #doctest:+ELLIPSIS
...
>>> glove = Glove(word_vectorizer.token2id, word_vectorizer.oov_token)
>>> glove.load(vectors_file.name)  #doctest:+ELLIPSIS
...
>>> embedding_layer = glove.get_embedding_layer(input_length=7)
>>> assert embedding_layer.input_dim = word_vectorizer.num_tokens
```
## Documentation
You can read about all methods and attributes of the classes in the `doc`
directory.

## Examples
Working examples are available as Jupyter notebooks in the `notebooks` 
directory.