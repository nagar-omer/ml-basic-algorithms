from math import log2
from data_loader import DataSet
from measures import get_measures

MAX = "__MAX__"
COUNT = "__COUNT__"


class DecisionNode:
    """
    class DecisionNode represents a single node in the decision tree
    """
    def __init__(self, parent={"root": None}, attribute=None, branches=None, label=None, is_leaf=False):
        """
        constructor
        :param parent: node parent { name: DecisionNode object}, default=root
        :param attribute: node_id / attribute_id
        :param branches: dictionary of children { attribute_val (edge_val) : next_child (child at end of node)}
        :param label: ONLY FOR LEAF: label (y_hat) ("yes/"no")
        :param is_leaf: True if node is list and false otherwise
        """
        self._parent, self._attribute,  self._branches, self._label, self._is_leaf = parent, attribute, \
                                                                                     branches, label, is_leaf

    # some properties for easy access
    @property
    def id(self):
        return self._attribute

    @property
    def branches(self):
        return self._branches

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def parent(self):
        return self._parent

    @property
    def label(self):
        return self._label

    def set_parent(self, parent_name, parent_node):
        """
        set DecisionNode obj as parent
        :param parent_name: parent attribute_id
        :param parent_node: parent attribute DecisionNode object
        """
        self._parent = {parent_name: parent_node}

    def get_child(self, value):
        """
        get next child with respect to feature-value
        :param value: possible feature-value
        :return: corresponding DecisionNode object
        """
        return self.branches[value]

    def to_string(self):
        """
        :return: string of sub_tree stating from current node as root
        """
        str_build = []
        for val, child in sorted(self._branches.items(), key=lambda x: x[0]):
            if child.is_leaf:
                # if next child is a leaf node than print its label
                str_build.append(("" if "root" in self.parent else "|") + self.id + "=" + val + ":" + child.label)
            else:
                # print edge
                str_build.append(("" if "root" in self.parent else "|") + self.id + "=" + val)
                # print edge sub tree
                str_build += ["\t" + s for s in child.to_string()]
        return str_build


class DecisionTree:
    """
    Decision Tree implements ID3 algorithm
    """
    def __init__(self, index_to_attribute):
        """
        constructor
        :param index_to_attribute: an ordered list with attributes-id as in input label
        """
        self._index_to_attribute = index_to_attribute
        self._attribute_to_index = {att: i for i, att in enumerate(self._index_to_attribute)}

        # filled after fit is called
        self._attribute_values = None
        # root node
        self._tree = None

    def __str__(self):
        return "" if self._tree is None else "\n".join(self._tree.to_string())

    def _get_attribute_values(self, ds):
        """
        :param ds: train-set
        :return: a map { attribute_id: [ possible values ] } according to train-set
        """
        map_att_val = {att: set() for att in self._index_to_attribute}
        # loop examples
        for vec, label in ds.data:
            # loop vector values / attributes
            for i, att in enumerate(self._index_to_attribute):
                map_att_val[att].add(vec[i])
        return map_att_val

    def _mode(self, examples, mode_typ=MAX):
        """
        count labels and returns max/count according to input
        :param examples: list of examples [ ... (vec, label) ... ]
        :param mode_typ: MAX / CONT
        :return:
        """
        count = {}
        for _, label in examples:
            count[label] = count.get(label, 0) + 1
        if mode_typ == MAX:
            return max(count.items(), key=lambda x: x[1])
        if mode_typ == COUNT:
            return count

    def _entropy(self, count):
        """
        :param count: { event: occurrences }
        :return: SUM_(i in events){ -p_i log(p_i) }
        """
        total = sum(list(count.values()))
        return sum(-p / (total + 1e-5) * log2(p / (total + 1e-5)) for p in list(count.values()))

    def _argmax_gain(self, examples, attributes):
        """
        argmax_att { Entropy(S_all) - Sum (i in att-values) [ |S_i| / |S_all| * Entropy(S_i)] } (Eq 1*) =
        argmin_att { Sum (i in att-values) [ |S_i| * Entropy(S_i)] } (Eq 2*)

        :param examples: list of examples [ ... (vec, label) ... ]
        :param attributes: list of attributes not filtered by yet
        :return: argmin according to Eq 2*
        """
        vec_sub_all_entropy = {}
        # calc for each attribute
        for att in attributes:
            att_idx = self._attribute_to_index[att]

            att_entropy_vec = []
            # for each value, count #(v[attribute] = val ^ label ) -> calculate Entropy(count)
            for val in self._attribute_values[att]:
                count = {}
                for vec, label in examples:
                    if vec[att_idx] == val:
                        count[label] = count.get(label, 0) + 1

                att_entropy_vec.append(self._entropy(count) * sum(list(count.values())))
            # Sum (i in att-values) [|S_i| * Entropy(S_i)]
            vec_sub_all_entropy[att] = (sum(att_entropy_vec))
        # return argmin
        return min(vec_sub_all_entropy.items(), key=lambda x: x[1])[0]

    def fit(self, ds):
        """
        fit model according to data-set
        """
        # examples=ALL, attributes=ALL, default=Majority-Vote(ALL)
        self._attribute_values = self._get_attribute_values(ds)
        examples = [(v, l) for v, l in ds]
        attributes = self._index_to_attribute
        max_count_attribute, count_attribute = self._mode(examples, mode_typ=MAX)
        # get tree
        self._tree = self._dtl(examples, attributes, max_count_attribute)
        return self._tree

    def _dtl(self, examples, attributes, default):
        """
        implementation of ID3 algorithm
        :param examples: examples left in current node
        :param attributes: attributes not filtered by yet
        :param default: default parent label
        :return: decision subtree (DecisionNode root)
        """
        # no examples left -> return parent default
        if not examples:
            return DecisionNode(label=default, is_leaf=True)
        count = self._mode(examples, mode_typ=COUNT)
        max_count_attribute, count_attribute = max(count.items(), key=lambda x: x[1])
        # only one possible label -> return the label || no attributes left to filter by -> return Majority-Vote
        if len(count) == 1 or not attributes:
            return DecisionNode(label=max_count_attribute, is_leaf=True)

        # best attribute by gain
        best_att = self._argmax_gain(examples, attributes)
        att_idx = self._attribute_to_index[best_att]
        branches = {}
        # for possible value of the attribute
        for val in self._attribute_values[best_att]:
            # get sub-tree by 1. filter_examples by current value
            #                 2. attributes - best attribute
            #                 3. Majority-Vote(current examples)
            examples_with_val = [(vec, label) for vec, label in examples if vec[att_idx] == val]
            attributes_left = [a for a in attributes if a != best_att]
            branches[val] = self._dtl(examples_with_val, attributes_left, self._mode(examples, mode_typ=MAX)[0])
        # build subtree and set parent
        best_att_sub_tree = DecisionNode(attribute=best_att, branches=branches)
        for val, node in branches.items():
            node.set_parent(best_att, best_att_sub_tree)
        return best_att_sub_tree

    def predict(self, test_ds: DataSet):
        """
        predict label of examples
        :return: ordered prediction and true labels lists
        """
        # if tree wasn't built yet
        if self._tree is None:
            print("fit on a train set first")
            return

        true, predict = [], []
        # for each example
        for vec, label in test_ds:
            # start iin root node and go down until leaf reached
            curr_node = self._tree
            while not curr_node.is_leaf:
                curr_node = curr_node.get_child(vec[self._attribute_to_index[curr_node.id]])

            # get prediction and label
            true.append(label)
            predict.append(curr_node.label)
        return predict, true


def check_dtl():
    ds_ = DataSet("dataset.txt")
    dtl_ = DecisionTree(ds_.header)
    tree_ = dtl_.fit(ds_)
    predict, true = dtl_.predict(ds_)

    TP, TN, FP, FN, acc, recall, precision, f1 = get_measures(predict, true)
    f = open("tree.txt", "wt")
    f.write(str(dtl_))
    print("accuracy:", acc, "\n"
          "recall:", recall, "\n"
          "precision:", precision, "\n"
          "f1:", f1, "\n")
    e = 0


if __name__ == '__main__':
    check_dtl()
