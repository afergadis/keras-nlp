import numpy as np
from sklearn.metrics import classification_report as cr


def strip_pad(y, offsets=None, pad_value=0):
    """ For a give array of label vectors (`y`), remove the pad values.

    Given that for a sequence labeling the labels `y` are padded with
    `pad_value`, we want to remove those and keep the offset of the first "true"
    label in the sequence.

    Parameters
    ----------
    y: 2D np.array
        The array of label vectors. Each row is the padded sequence labels.
    offsets: 1D np.array, default=None
        If offsets is `None`, then for each row in `y`, remove the pad
        values and mark the offset of the first non padded value. Otherwise,
        for each row in `y`, we remove the values that are before `offset`.
    pad_value: int, default=0
        The value used to pad the labels sequences.

    Returns
    -------
    tuple
        A tuple `(striped_y, offsets)`, where `striped_y` is the `y` without the
        pad values and `offset` the offsets of the first non padded value for
        each sequence.

    Examples
    --------
    >>> y = np.array([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]])
    >>> # The pad value is 0 and for the 1st sequence, the first non zero label
    >>> # is on id 2, and for the 2nd sequence on id 1.
    >>> striped_y, offsets = strip_pad(y)
    >>> print(striped_y)
    [[1, 1, 2], [1, 1, 2, 2]]
    >>> print(offsets)
    [2, 1]
    """
    if offsets is None:
        # For every sentence find the first non `pad_value` label.
        offsets = [np.where(v > pad_value)[0][0] for v in y]
    # Remove PAD.
    striped_y = [v[offset:].tolist() for v, offset in zip(y, offsets)]
    return striped_y, offsets


def flatten(y_true, y_pred, pad_value=0):
    """ Remove `pad_value` from 2D arrays of sequence labels and flatten them
    to 1D.

    Parameters
    ----------
    y_true: 2D array-like
        A 2D array of sequence labels representing the true/gold labels.

    y_pred: 2D array-like
        A 2D array of sequence labels representing the predicted labels.

    pad_value : int
        The value used to pad sequences. This value will be removed from
        `y_true` and `y_pred`.

    Examples
    --------
    >>> y_true = np.array([[0, 0, 1, 1, 2]])
    >>> y_pred = np.array([[0, 1, 1, 2, 2]])
    >>> y_gold, y_hat = flatten(y_true, y_pred)
    >>> print(y_gold)  # Will print from the first non zero item.
    [1, 1, 2]
    >>> print(y_hat)  # Will print from the 1st non zero item of y_gold list.
    [1, 2, 2]
    """
    y_true, offsets = strip_pad(y_true, pad_value=pad_value)
    # Remove PAD from y_pred.
    y_pred, _ = strip_pad(y_pred, offsets, pad_value)
    # Flatten sequences.
    y_gold = [c for row in y_true for c in row]
    y_hat = [c for row in y_pred for c in row]

    return y_gold, y_hat


def classification_report(y_true, y_pred, **kwargs):
    """ Classification report for sequence labeling.

    Parameters
    ----------
    y_true: 2D np.array or list of lists
        The true/gold sequence labels.

    y_pred: 2D np.array of list of lists
        The prediction labels

    **kwargs:
        The parameters of the sklearn.metrics.classification_report function.
    """
    y_gold, y_hat = flatten(y_true, y_pred)
    return cr(y_gold, y_hat, **kwargs)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
