import numpy as np

__author__ = 'feldsherov'

class DecisionTreeRegressor():
    """
    DecisionTreeRegressor class implements DecisionTree model for regressing problem
    """

    class DecisionTreeNode:
        def __init__(self):
            pass

    def __init__(self, max_depth=10, min_list_size=2, min_list_variance=1e-2, min_branch_size=0.3):
        self.max_depth = max_depth
        self.min_list_size = min_list_size
        self.root = None
        self.count_features = None
        self.train_set_size = None
        self.min_list_variance = min_list_variance
        self.min_branch_size = min_branch_size

    def __train_tree(self, root, x, y, curr_depth=0):
        variance = y.var()
        if curr_depth == self.max_depth or x.shape[0] <= self.min_list_size or variance < self.min_list_variance:
            root.list_result = y.mean()
            return

        feature_id, predicate, border_value = self.__get_optimal_split(x, y)

        root.left = DecisionTreeRegressor.DecisionTreeNode()
        root.right = DecisionTreeRegressor.DecisionTreeNode()
        root.predicate = predicate
        root.feature_id = feature_id
        root.border_value = border_value

        actual_objects_left = np.array([predicate(el) for el in x[::, feature_id]])

        self.__train_tree(root.left, x[actual_objects_left], y[actual_objects_left], curr_depth + 1)
        self.__train_tree(root.right, x[-actual_objects_left], y[-actual_objects_left], curr_depth + 1)

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self.root = DecisionTreeRegressor.DecisionTreeNode()
        self.count_features = x.shape[1]
        self.train_set_size = x.shape[0]
        self.__train_tree(self.root, x, y)

    def predict(self, x):
        x = np.array(x)
        return np.array([self.__get_one_prediction(self.root, elem) for elem in x])

    def __get_one_prediction(self, root, elem):
        if "left" not in root.__dict__ and "right" not in root.__dict__:
            return root.list_result

        if root.predicate(elem[root.feature_id]):
            return self.__get_one_prediction(root.left, elem)
        else:
            return self.__get_one_prediction(root.right, elem)

    def __get_optimal_split(self, x, y):
        current_actual_set_size = x.shape[0]

        sum_y = y.sum()
        sum_sq_y = (y ** 2).sum()

        sort_order = x.argsort(axis=0)
        optimal_feature_id, optimal_elem_id, optimal_variance = 0, 0, float("inf")

        for feature_id in range(self.count_features):
            sum_sq_left = np.zeros(current_actual_set_size)
            sum_left = np.zeros(current_actual_set_size)
            for elem_id in range(1, current_actual_set_size):
                sum_left[elem_id] = sum_left[elem_id - 1] + \
                                                y[sort_order[elem_id - 1][feature_id]]
                sum_sq_left[elem_id] = sum_sq_left[elem_id - 1] + \
                                        y[sort_order[elem_id - 1][feature_id]] ** 2

                current_variance = \
                    (
                        sum_sq_left[elem_id] / elem_id -
                        (sum_left[elem_id] / elem_id) ** 2
                    ) + \
                    (
                        (sum_sq_y - sum_sq_left[elem_id]) / (current_actual_set_size - elem_id) -
                        ((sum_y - sum_left[elem_id]) / (current_actual_set_size - elem_id)) ** 2
                    )

                if x[sort_order[elem_id - 1][feature_id]][feature_id] != x[sort_order[elem_id][feature_id]][feature_id]\
                        and current_variance < optimal_variance \
                        and current_actual_set_size * self.min_branch_size < elem_id < current_actual_set_size * (
                            1 - self.min_branch_size):
                    optimal_feature_id, optimal_elem_id, optimal_variance = feature_id, elem_id, current_variance

        return (optimal_feature_id,
                lambda a: a < x[sort_order[optimal_elem_id][optimal_feature_id]][optimal_feature_id],
                x[sort_order[optimal_elem_id][optimal_feature_id]][optimal_feature_id])