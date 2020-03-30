import re


def sent_tokenize(
        text,
        subst="\n",
        regex="(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
):
    """
    A simple sentence tokenizer based on regular expressions to find the
    sentence boundaries.

    Parameters
    ----------
    text : str
        The text to tokenize.

    subst : str
        The substitution character that defines a sentence boundary.

    regex : str
        The regular expresion to use in order to find sentence boundaries.

    Returns
    -------
    list
        A list with the sentences of the text.

    See Also
    --------
    `https://stackoverflow.com/a/25736082/1143894`_

    Examples
    --------
    >>> doc = "MI patients had 18% higher plasma levels of MAp44 "\
    "(IQR 11-25%) as compared to the healthy control group (p < 0.001). "\
    "However, neither salvage index (Spearman rho -0.1, p = 0.28) " \
    "nor final infarct size (Spearman rho 0.02, p = 0.83) correlated "\
    "with plasma levels of MAp44."
    >>> sents = sent_tokenize(doc)
    >>> print(len(sents))
    2
    """
    substitutions = re.sub(regex, subst, text, 0, re.MULTILINE)
    return list(substitutions.split(subst))


def word_tokenize(text, regex="\W+|\t|\n"):
    """
    A simple word tokenizer to tokenize text on the giver regular expression.

    Parameters
    ----------
    text : str
        The text to tokenize.

    regex : str
        The regular expresion to use in order to split the `text`.

    Returns
    -------
    list
        A list with the words in the text. Separators are trimmed.

    Examples
    --------
    >>> sentence = 'A    sentence\\tto\\nbe   tokenized\\n'
    >>> word_tokenize(sentence)
    ['A', 'sentence', 'to', 'be', 'tokenized']
    """
    tokens = re.split(regex, text)
    if len(tokens[-1]) == 0:
        tokens.pop()

    return tokens


class SentenceSplitter:
    """
    Splitting sentences using some heuristics that help to prevent wrong
    segmentation especially in scientific papers.

    Examples
    --------
    >>> doc = "MI patients had 18% higher plasma levels of MAp44 "\
    "(IQR 11-25%) as compared to the healthy control group (p < 0.001). "\
    "However, neither salvage index (Spearman rho -0.1, p = 0.28) " \
    "nor final infarct size (Spearman rho 0.02, p = 0.83) correlated "\
    "with plasma levels of MAp44."
    >>> ss = SentenceSplitter()  # Use the module's sent_tokenize function.
    >>> print(len(ss.tokenize(doc)))
    2
    """
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = sent_tokenize
        else:
            self.tokenizer = tokenizer

    @staticmethod
    def _first_alpha_is_upper(sent):
        for c in sent:
            if c.isalpha():
                if c.isupper():
                    return True

        return False

    @staticmethod
    def _ends_with_special(sent):
        sent = sent.lower()
        ind = [
            item.end() for item in re.finditer(
                r'[\W\s]sp.|[\W\s]nos.|[\W\s]figs.|[\W\s]sp.[\W\s]no.|'
                r'[\W\s][vols.|[\W\s]cv.|[\W\s]fig.|[\W\s]e..|'
                r'[\W\s]et[\W\s]al.|[\W\s]i.e.|[\W\s]p.p.m.|[\W\s]cf.|'
                r'[\W\s]n.a.', sent)
        ]
        if len(ind) == 0:
            return False
        else:
            ind = max(ind)
            if len(sent) == ind:
                return True
            else:
                return False

    def _split_sentences(self, text):
        sents = [l.strip() for l in self.tokenizer(text)]
        ret = []
        i = 0
        while i < len(sents):
            sent = sents[i]
            while (i + 1) < len(sents) and (
                    self._ends_with_special(sent)
                    or not self._first_alpha_is_upper(sents[i + 1])):
                sent += ' ' + sents[i + 1]
                i += 1
            ret.append(sent.replace('\n', ' ').strip())
            i += 1
        return ret

    def tokenize(self, text: str):
        """
        Tokenize a text to its sentences.

        Parameters
        ----------
        text : str
            The text to tokenize.

        Returns
        -------
        list
            A list of sentences.
        """
        sents = []
        subtext = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
        if len(subtext) > 0:
            ss = self._split_sentences(subtext)
            sents.extend([s for s in ss if (len(s.strip()) > 0)])
        if len(sents[-1]) == 0:
            sents = sents[:-1]
        return sents


if __name__ == '__main__':
    import doctest

    doctest.testmod()
