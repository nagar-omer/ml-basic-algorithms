from data_loader import DataSet
from measures import get_measures


class Knn:
    """
    class KNN implements K nearest neighbours algorithm
    """
    def __init__(self, k):
        """
        :param k: number of neighbours to check
        """
        self._k = k
        # filled after fit is called
        self._train_ds = None

    def fit(self, ds: DataSet):
        """
        not much to do, make ds as potential neighbours
        :param ds: train data-set
        """
        self._train_ds = ds
        return ds

    def hamming_distance(self, str_1, str_2):
        """
        :param str_1, str_2: two list of characters
        :return: number of places were list_1 != list_2
        """
        return sum([1 for char_1, char_2 in zip(str_1, str_2) if char_1 != char_2])

    def predict(self, test_ds: DataSet):
        # if there are no potential neighbours
        if self._train_ds is None:
            print("fit on a train set first")
            return

        true, predict = [], []
        # for each example
        for vec, label in test_ds:
            # get K nearest neighbours
            nearest_k = sorted([(self.hamming_distance(vec, train_vec), train_label) for
                                train_vec, train_label in self._train_ds],  key=lambda x: x[0])[:self._k]
            # check Majority-Vote among nearest neighbours
            count = {}
            for _, l in nearest_k:
                count[l] = count.get(l, 0) + 1
            pred = max(count.items(), key=lambda x: x[1])[0]

            # return prediction and true-label
            predict.append(pred)
            true.append(label)
        return predict, true


def check_knn():
    ds_ = DataSet("dataset.txt")
    knn_ = Knn(5)
    knn_.fit(ds_)
    predict, true = knn_.predict(ds_)

    TP, TN, FP, FN, acc, recall, precision, f1 = get_measures(predict, true)
    print("accuracy:", acc, "\n"
          "recall:", recall, "\n"
          "precision:", precision, "\n"
          "f1:", f1, "\n")


if __name__ == '__main__':
    check_knn()
