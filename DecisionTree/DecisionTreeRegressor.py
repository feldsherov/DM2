import numpy as np


class DecisionTreeRegressor():
    """
    DecisionTreeRegressor class implements DecisionTree model for regressing problem
    """

    class DecisionTreeNode():
        def __init__(self):
            pass

    def __init__(self, max_depth=10, min_list_size=2, min_list_variance=1e-2):
        self.max_depth = max_depth
        self.min_list_size = min_list_size
        self.root = None
        self.count_features = None
        self.train_set_size = None
        self.min_list_variance = min_list_variance

    def train_tree(self, root, x, y, curr_depth=0):
        variance = ((y - (y.sum() / y.size))**2).sum() / y.size
        if curr_depth == self.max_depth or x.shape[0] <= self.min_list_size or variance < self.min_list_variance:
            root.list_result = y.sum() / y.size
            return

        feature_id, predicate = self.get_optimal_split(x, y)

        root.left = DecisionTreeRegressor.DecisionTreeNode()
        root.right = DecisionTreeRegressor.DecisionTreeNode()
        root.predicate = predicate
        root.feature = feature_id

        actual_objects_left = np.zeros(x.shape[0], dtype=bool)
        for i, elem in enumerate(x):
            actual_objects_left = predicate(elem[i])

        self.train_tree(root.left, x[actual_objects_left], y[actual_objects_left], curr_depth + 1)
        self.train_tree(root.right, x[not actual_objects_left], y[not actual_objects_left], curr_depth + 1)

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self.root = DecisionTreeRegressor.DecisionTreeNode()
        self.count_features = x.shape[1]
        self.train_set_size = x.shape[0]
        self.train_tree(self.root, x, y)

    def get_optimal_split(self, x, y):
        current_actual_set_size = x.shape[0]

        sum_y = x.sum()
        sum_sq_y = (y ** 2).sum()

        sort_order = x.argsort(axis=0)
        optimal_feature_id, optimal_elem_id, optimal_variance = \
            0, 0, sum_sq_y / current_actual_set_size - (sum_y / current_actual_set_size) ** 2

        for feature_id in range(self.count_features):
            sum_sq_left = np.zeros(current_actual_set_size)
            sum_left = np.zeros(current_actual_set_size)
            for elem_id in range(1, current_actual_set_size):
                sum_left[elem_id] = sum_left[elem_id - 1] + \
                                                x[sort_order[elem_id][feature_id]][feature_id]
                sum_sq_left[elem_id] = sum_sq_left[elem_id - 1] + \
                                        x[sort_order[elem_id][feature_id]][feature_id] ** 2

                current_variance = \
                    (
                        sum_sq_left[elem_id] / (elem_id + 1) -
                        (sum_left[elem_id] / (elem_id + 1)) ** 2
                    ) + \
                    (
                        (sum_sq_y - sum_sq_left[elem_id]) / (current_actual_set_size - elem_id - 1) -
                        ((sum_y - sum_left[elem_id]) / (current_actual_set_size - elem_id - 1)) ** 2
                    )

                if current_variance < optimal_variance:
                    optimal_feature_id, optimal_elem_id, optimal_variance = feature_id, elem_id, current_variance

        return optimal_feature_id, lambda a: a < x[sort_order[elem_id][feature_id]][feature_id]