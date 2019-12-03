import logging
from keras_nlp.preprocessing.text import (CharVectorizer, SentCharVectorizer,
                                          WordVectorizer, SentWordVectorizer)
from keras_nlp.preprocessing.segment import SentenceSplitter
from keras_nlp.data import load_dataset, Dataset
from keras_nlp.mappings import Glove, W2V

logging.basicConfig(
    level=logging.INFO,
    datefmt='%y-%b-%d %H:%M:%S',
    format='%(asctime)s [%(levelname)-8s:%(name)-12s] - %(message)s')
