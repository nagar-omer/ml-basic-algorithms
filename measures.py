POSITIVE, NEGATIVE = "yes", "no"


def get_measures(pred, true):
    """
    calculate measures for binary classifications
    :param pred: predictions
    :param true: true labels
    :return: success measures TP, TN, FP, FN, acc, recall, precision, f1
    """
    # count True-Positive, True-Negative, False-Positive and False-Negative
    TP, TN, FP, FN = 0, 0, 0, 0
    for p, t in zip(pred, true):
        if p == t:
            if t == POSITIVE:  # true positive
                TP += 1
            else:  # true negative
                TN += 1
        else:
            if p == POSITIVE:  # false positive
                FP += 1
            else:  # false negative
                FN += 1
    # calculate measures according to https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    acc = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN + 1e-5)
    f1 = 2 * TP / (2 * TP + FP + FN + 1e-5)
    precision = TP / (TP + FP + 1e-5)
    return TP, TN, FP, FN, acc, recall, precision, f1
