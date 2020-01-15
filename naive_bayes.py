from data_loader import DataSet
from measures import get_measures


class NaiveBayes:
    """
    class NaiveBayes implements Naive Bayes algorithm
    """
    def __init__(self, index_to_attribute):
        """
        constructor
        :param index_to_attribute: an ordered list with attributes-id as in input label
        """
        self._index_to_attribute = index_to_attribute
        self._attribute_to_index = {att: i for i, att in enumerate(self._index_to_attribute)}
        # P(Attribute=value | Label=y), P(Label=y)
        self._prob_feature_given_label, self._prob_labels = None, None

    def fit(self, ds: DataSet):
        """
        calculate probabilities according to train-set
        :return: P(Attribute=value | Label=y), P(Label=y)
        """
        # { attribute: possible-values }
        feature_to_values = {att: set() for att in self._index_to_attribute}
        # count_labels                  #(Label=y)
        # count_features_and_label      #(Attribute=value /\ Label=y)
        count_labels, count_features_and_label = {}, {}

        # go over the data and count the above
        for vec, label in ds:
            # #(Label=y) += 1
            count_labels[label] = count_labels.get(label, 0) + 1
            for i, att in enumerate(self._index_to_attribute):
                feature_to_values[att].add(vec[i])
                # (Attribute=value /\ Label=y) += 1
                count_features_and_label[(att, vec[self._attribute_to_index[att]], label)] = \
                    count_features_and_label.get((att, vec[self._attribute_to_index[att]], label), 0) + 1

        # # { attribute: | possible-values | }
        feature_to_values = {att: len(values_set) for att, values_set in feature_to_values.items()}

        # P(Label=y) = #(Label=y) / #(ALL)
        self._prob_labels = {label: count / len(ds) for label, count in count_labels.items()}
        #                                    #(Attribute=value /\ Label=y) + 1
        # P(Attribute=value | Label=y) =  ---------------------------------------
        #                                    #(Label=y) + | Attribute-Values |
        self._prob_feature_given_label = {(att, val, label): (count + 1) /
                                                             (count_labels[label] + feature_to_values[att])
                                          for (att, val, label), count in count_features_and_label.items()}
        return self._prob_labels, self._prob_feature_given_label

    def _multiply(self, vec):
        """
        cant use any packages so ... :\
        """
        res = 1
        for v in vec:
            res *= v
        return res

    def predict(self, test_ds: DataSet):
        # if no probabilities were calculated
        if self._prob_labels is None:
            print("fit on a train set first")
            return

        true, predict = [], []
        # for each example
        for vec, true_label in test_ds:
            score_labels = {}
            for label in self._prob_labels:
                # Given Vec = [A_1=v_1, A_2=v_2, .... A_n=v_n], Label=y
                # calculate for each label y, and return argmax
                #    ______
                # (   |  |   P(A_i=v_i | Label=y) )  *  P(Label=y)
                #       i
                score_labels[label] = self._multiply([self._prob_feature_given_label[(att, vec[i], label)]
                                                     for i, att in enumerate(self._index_to_attribute)] +
                                                     [self._prob_labels[label]])
            pred = max(score_labels.items(), key=lambda x: x[1])[0]

            # return prediction and label
            predict.append(pred)
            true.append(true_label)

        return predict, true


def check_naive_bayes():
    ds_ = DataSet("dataset.txt")
    naive_bayes_ = NaiveBayes(ds_.header)
    naive_bayes_.fit(ds_)
    predict, true = naive_bayes_.predict(ds_)

    TP, TN, FP, FN, acc, recall, precision, f1 = get_measures(predict, true)
    print("accuracy:", acc, "\n"
          "recall:", recall, "\n"
          "precision:", precision, "\n"
          "f1:", f1, "\n")


if __name__ == '__main__':
    check_naive_bayes()
