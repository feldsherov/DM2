import numpy as np
import sys

__author__ = 'Svyatoslav Feldsherov'


class DecisionTreeRegressor():
    """
    DecisionTreeRegressor class implements DecisionTree model for regressing problem
    """

    class DecisionTreeNode:
        """
        Object implements TreeNode. Empty class for storage purposes.
        """

        def __init__(self):
            pass

    def __init__(self, max_depth=50, min_list_size=2, min_list_variance=1e-10):
        """
        :param max_depth: maximum depth of the tree
        :param min_list_size: minimum list size for continuation splitting
        :param min_list_variance: minimum list variance for continuation splitting
        :return: None
        """
        self.max_depth = max_depth
        self.min_list_size = min_list_size
        self.root = None
        self.count_features = None
        self.train_set_size = None
        self.min_list_variance = min_list_variance

    def __visualize_rec(self, root, depth):
        """
        print ro stderr tree
        :param root: root of current subtree
        :param depth:
        :return:
        """

        if "left" not in root.__dict__ and "right" not in root.__dict__:
            print >>sys.stderr, "    "*depth, "list_result=%s" % root.list_result
            return

        self.__visualize_rec(root.left, depth+1)
        print >>sys.stderr, "    "*depth, "id=%s split_el=%s" % (root.feature_id, root.split_el)
        self.__visualize_rec(root.right, depth+1)

    def visualize(self):
        self.__visualize_rec(self.root, 0)

    def __train_tree(self, root, x, y, curr_depth=0):
        """
        recursive function for train tree
        :param root: the root of current subtree
        :param x: 2D array of features x[object][feature]
        :param y: target variable
        :param curr_depth: depth of current Node
        :return: None
        """

        # check stop conditions
        variance = y.var()
        if curr_depth == self.max_depth or x.shape[0] <= self.min_list_size or variance < self.min_list_variance:
            root.list_result = y.mean()
            return

        # finding optimal split
        feature_id, predicate, split_el = self.__get_optimal_split(x, y)

        # initialization of current node
        root.left = DecisionTreeRegressor.DecisionTreeNode()
        root.right = DecisionTreeRegressor.DecisionTreeNode()
        root.predicate = predicate
        root.feature_id = feature_id
        root.split_el = split_el

        # boolean array: is object belong to left child
        actual_objects_left = np.array([predicate(el) for el in x[::, feature_id]])

        # recursive calls for train subtrees
        self.__train_tree(root.left, x[actual_objects_left], y[actual_objects_left], curr_depth + 1)
        self.__train_tree(root.right, x[-actual_objects_left], y[-actual_objects_left], curr_depth + 1)

    def fit(self, x, y):
        """
        :param x: 2D array of features x[object][feature]
        :param y: target variable
        :return: None
        """
        x = np.array(x)
        y = np.array(y)
        self.root = DecisionTreeRegressor.DecisionTreeNode()
        self.count_features = x.shape[1]
        self.train_set_size = x.shape[0]
        self.__train_tree(self.root, x, y)

    def predict(self, x):
        """
        :param x: 2D array of features x[object][feature]
        :return: numpy.arrays of predictions
        """
        x = np.array(x)
        return np.array([self.__get_one_prediction(self.root, elem) for elem in x])

    def get_features_profit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        profit = np.zeros(x.shape[1])
        self.__get_feature_profit_by_node(self.root, x, y, profit)
        return profit

    def __get_feature_profit_by_node(self, root, x, y, profit):
        """
        recursive function for calculating profit
        :param root: the root of current subtree
        :param x: 2D array of features x[object][feature]
        :param y: target variable
        :param profit: 1D array for output
        :return:
        """

        # check for current node is list
        if "left" not in root.__dict__ and "right" not in root.__dict__:
            return root.list_result

        predicate = root.predicate
        feature_id = root.feature_id
        actual_objects_left = np.array([predicate(el) for el in x[::, feature_id]])

        actual_left = y[actual_objects_left]
        actual_right = y[-actual_objects_left]

        profit[feature_id] += y.var() * y.size -\
                              actual_left.var() * actual_left.size - actual_right.var() * actual_right.size

        self.__get_feature_profit_by_node(root.left, x[actual_objects_left], actual_left, profit)
        self.__get_feature_profit_by_node(root.right, x[-actual_objects_left], actual_right, profit)

    def __get_one_prediction(self, root, elem):
        """
        :param root: the root of current subtree
        :param elem: vector of features of current object
        :return: float -- prediction for vector of features X
        """

        # check for current node is list
        if "left" not in root.__dict__ and "right" not in root.__dict__:
            return root.list_result

        if root.predicate(elem[root.feature_id]):
            return self.__get_one_prediction(root.left, elem)
        else:
            return self.__get_one_prediction(root.right, elem)

    def __get_optimal_split(self, x, y):
        """
        :param x:
        :param y:
        :return: tuple (int, function)
        feature_id -- id of feature for split
        predicate -- true/false function for decision left/right child
        """
        current_actual_set_size = x.shape[0]

        sum_y = y.sum()
        sum_sq_y = (y ** 2).sum()

        sort_order = x.argsort(axis=0)
        optimal_feature_id, optimal_elem_id, loss = 0, 0, float("inf")

        for feature_id in range(self.count_features):
            # arrays of prefix sums for dynamic calculation of variance
            sum_sq_left = np.zeros(current_actual_set_size)  # sum of squares of y[i]
            sum_left = np.zeros(current_actual_set_size)  # sum of y[i]

            for elem_id in range(1, current_actual_set_size):
                # recalculating of prefix sums (dynamic recalculation)
                sum_left[elem_id] = sum_left[elem_id - 1] + \
                                    y[sort_order[elem_id - 1][feature_id]]
                sum_sq_left[elem_id] = sum_sq_left[elem_id - 1] + \
                                       y[sort_order[elem_id - 1][feature_id]] ** 2

                # calculation of variance by formula D(x) = E(x**2) - E(x)**2
                variance_left = sum_sq_left[elem_id] / elem_id - (sum_left[elem_id] / elem_id) ** 2
                variance_right = (sum_sq_y - sum_sq_left[elem_id]) / (current_actual_set_size - elem_id) - \
                                 ((sum_y - sum_left[elem_id]) / (current_actual_set_size - elem_id)) ** 2

                # calculating loss for current split
                current_variance = elem_id * variance_left + (current_actual_set_size - elem_id) * variance_right

                if x[sort_order[elem_id - 1][feature_id]][feature_id] != \
                        x[sort_order[elem_id][feature_id]][feature_id] and current_variance < loss:
                    optimal_feature_id, optimal_elem_id, loss = feature_id, elem_id, current_variance

        pred_split_val = x[sort_order[optimal_elem_id - 1][optimal_feature_id]][optimal_feature_id]
        next_split_val = x[sort_order[optimal_elem_id][optimal_feature_id]][optimal_feature_id]
        act_split_val = (pred_split_val + next_split_val) / 2
        return (optimal_feature_id,
                lambda a: a <= act_split_val,
                act_split_val)