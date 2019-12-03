from .data import Dataset


def load_dataset(file_path):
    """ Helper function to instantiate a `Dataset` object and use it load()
    function.

    Parameters
    ----------
    file_path : str
        The path to the file name of the saved Dataset file.

    Returns
    -------
    `data.Dataset`
    """
    return Dataset.load(file_path)
