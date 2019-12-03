import os


def load_imdb(imdb_dir):
    train_dir = os.path.join(imdb_dir, 'train')
    train_texts, train_labels = load_data(train_dir)
    test_dir = os.path.join(imdb_dir, 'test')
    test_texts, test_labels = load_data(test_dir)

    return (train_texts, train_labels), (test_texts, test_labels)


def load_data(data_dir):
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                labels.append(label_type)

    return texts, labels
