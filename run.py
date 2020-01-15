from data_loader import DataSet, k_fold
from decision_tree import DecisionTree
from knn import Knn
from measures import get_measures
from naive_bayes import NaiveBayes


TRAIN_FILE = ""
TEST_FILE = ""

KNN = "__KNN__"
ID3 = "__ID3__"
NAIVE_BAYES = "__NAIVE_BAYES__"


def get_model(model, arg):
    """
    :param model: model name
    :param arg: argument
    :return: model(arg)
    """
    models = {
        KNN: Knn,
        ID3: DecisionTree,
        NAIVE_BAYES: NaiveBayes
    }
    return models[model](arg)


def load_data(train_path, test_path):
    """
    load train and test data-sets by files path
    """
    return DataSet(train_path), DataSet(test_path)


def fit_predict(model, train, test):
    """
    fit model on train, predict on test and return accuracy
    """
    model.fit(train)
    predict, true = model.predict(test)
    TP, TN, FP, FN, acc, recall, precision, f1 = get_measures(predict, true)
    return acc


def cross_validation(path_data, model, k=5):
    """
    perform K-fold cross validation on model & data-set
    :return: average accuracy
    """
    ds = DataSet(path_data)
    arg = 5 if model == KNN else ds.header

    accuracies = []
    # fit K times + check accuracy
    for i, (train, test) in enumerate(k_fold(k, ds)):
        acc = fit_predict(get_model(model, arg), train, test)
        print("fold " + str(i) + ": accuracy=" + str(acc))
        accuracies.append(acc)

    # return average
    aggregate_acc = sum(accuracies) / len(accuracies)
    print("aggregate accuracy=" + str(aggregate_acc))
    return aggregate_acc


def evaluate_models_cross_validation(path_data, models=(ID3, KNN, NAIVE_BAYES)):
    """
    evaluate several models with 5-fold cross validation
    :param path_data: data-set
    :param models: tuple, models to evaluate
    """
    accuracies = []
    for model in models:
        print(model + "\n" + "="*20)
        acc = cross_validation(path_data, model)
        accuracies.append(acc)
        print("="*20 + "\n")

    # generate accuracies file
    with open("accuracy.txt", "wt") as f:
        f.write("\t".join([str(round(v, 2)) for v in accuracies]))


def evaluate_models_train_test(path_train, path_test, models=(ID3, KNN, NAIVE_BAYES)):
    """
    evaluate several models with given train and test
    :param models: tuple, models to evaluate
    """
    ds_train = DataSet(path_train)
    ds_test = DataSet(path_test)

    accuracies = []
    model_list = {model: get_model(model, 5 if model == KNN else ds_train.header) for model in models}

    # for each model -> fit train-set and predict test set
    for model_name in models:
        print(model_name + "\n" + "="*20)
        acc = fit_predict(model_list[model_name], ds_train, ds_test)
        accuracies.append(acc)
        print("accuracy=" + str(acc))
        print("="*20 + "\n")

    return str(model_list[ID3]) + "\n\n" + "\t".join([str(round(v, 2)) for v in accuracies])


if __name__ == '__main__':
    evaluate_models_cross_validation("dataset.txt")
