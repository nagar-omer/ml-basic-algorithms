from random import shuffle
import math


class DataSet:
    """
    class DataSet to load and iterate data
    """
    def __init__(self, data_path, ready=None):
        """
        :param data_path: path to data-file
        :param ready: tuple (header, data)
        """
        # option to load from ready data
        if ready is not None:
            self._header, self._data = ready
            return

        # load from file
        self._header, self._data = self._build_data(data_path)

    def _build_data(self, data_path):
        """
        read file
        :param data_path: path to data-file
        :return: headder - ordered list of attributes && data - list [ ...(vec, label)... ]
        """
        data = []
        f_data = open(data_path, "rt")
        header = f_data.readline().split()[:-1]
        for sample in f_data:
            sample = sample.split()
            vec = sample[:-1]
            label = sample[-1]
            data.append((vec, label))
        return header, data

    @property
    def data(self):
        return self._data

    @property
    def header(self):
        return self._header

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]


def k_fold(k, ds: DataSet, shuffle_data=False):
    """
    iterator
    split data to K chunks
    return K pairs ( test-set(from group i), train-set(from the rest) )

                 0          1          2         ...        N-1         N
    data = ||          |          |          |   ...    |          |          ||
    fold 2 ||  TRAIN   |  TRAIN   |  Test    |  TRAIN   |  TRAIN   |  TRAIN   ||

    :param k: number of groups
    :param ds: full data-set
    :param shuffle_data: if true than data is shuffled before split
    :return: pairs (train-set, test-set)
    """
    indices = list(range(len(ds)))
    if shuffle_data:
        shuffle(indices)

    # split examples to K groups
    len_group = int(math.ceil(len(indices) / k))
    groups = [indices[i:i + len_group] for i in range(0, len(indices), len_group)]

    for test_idx, test_group in enumerate(groups):
        train_idx = [i for i in range(k) if i != test_idx]

        # get train examples indices
        train_group = []
        for i in train_idx:
            train_group += groups[i]

        # build data-sets
        yield DataSet(".", ready=(ds.header, [ds.__getitem__(i) for i in train_group])), \
              DataSet(".", ready=(ds.header, [ds.__getitem__(i) for i in test_group]))


if __name__ == '__main__':
    ds_ = DataSet("dataset.txt")
    for i_ in range(len(ds_)):
        print(ds_.__getitem__(i_))
