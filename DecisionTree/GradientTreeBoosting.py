import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

from DecisionTreeRegressor import DecisionTreeRegressor

__author__ = 'Svyatoslav Feldsherov'


class GradientTreeBoosting:
    """
    Implements gradient boosting of decision trees
    """
    def __init__(self, count_steps=200, step=1e-2, max_tree_depth=8):
        """
        :param count_steps: count of steps in boosting (count of trees)
        :param step: multiplier for model prediction in model
        :param max_tree_depth: max depth of any tree in ensemble
        :return None:
        """
        self.count_steps = count_steps
        self.step = step
        self.max_tree_depth = max_tree_depth
        self.coefficients = None
        self.models = None

    def fit(self, x, y):
        """
        :param x: 2D array of features, x[object][feature]
        :param y: target variable
        :return:
        """
        x = np.array(x)
        y = np.array(y)

        models, coefficients = list(), list()
        models.append(DecisionTreeRegressor(max_depth=self.max_tree_depth))
        models[-1].fit(x, y)
        coefficients.append(self.step)

        current_model_predictions = coefficients[-1] * models[-1].predict(x)
        print >>sys.stderr, "My tree step: %d, mean squared error: %f" % (0, mean_squared_error(current_model_predictions, y))

        for i in range(1, self.count_steps):
            antigrad = 2*(y - current_model_predictions)
            models.append(DecisionTreeRegressor(max_depth=self.max_tree_depth))
            models[-1].fit(x, antigrad)
            coefficients.append(self.step)

            current_model_predictions += coefficients[-1] * models[-1].predict(x)

            print >>sys.stderr, "My tree step: %d, mean squared error: %f" % (i, mean_squared_error(current_model_predictions, y))
            # print >>sys.stderr, current_model_predictions

        self.models = models
        self.coefficients = coefficients

    def predict(self, x):
        """
        :param x: 2D array of features x[object][feature]
        :return: array -- predicted values (len(y) == x.shape[1] )
        """
        x = np.array(x)
        predictions = np.zeros(x.shape[0])
        for i, tree in enumerate(self.models):
            predictions += self.coefficients[i] * tree.predict(x)

        return predictions


class GradientTreeBoostingViaSklearnTree:
    """
    Implements gradient boosting of decision trees
    """
    def __init__(self, count_steps=200, step=1e-2, max_tree_depth=8):
        """
        :param count_steps: count of steps in boosting (count of trees)
        :param step: multiplier for model prediction in model
        :param max_tree_depth: max depth of any tree in ensemble
        :return None:
        """
        self.count_steps = count_steps
        self.step = step
        self.max_tree_depth = max_tree_depth
        self.coefficients = None
        self.models = None

    def fit(self, x, y):
        """
        :param x: 2D array of features, x[object][feature]
        :param y: target variable
        :return:
        """
        x = np.array(x)
        y = np.array(y)

        models, coefficients = list(), list()
        models.append(SklearnDecisionTreeRegressor(max_depth=self.max_tree_depth))
        models[-1].fit(x, y)
        coefficients.append(self.step)

        current_model_predictions = coefficients[-1] * models[-1].predict(x)
        print >>sys.stderr, "Sklearn tree step: %d, mean squared error: %f" % (0, mean_squared_error(current_model_predictions, y))

        for i in range(1, self.count_steps):
            antigrad = 2*(y - current_model_predictions)
            models.append(SklearnDecisionTreeRegressor(max_depth=self.max_tree_depth))
            models[-1].fit(x, antigrad)
            coefficients.append(self.step)

            current_model_predictions += coefficients[-1] * models[-1].predict(x)

            print >>sys.stderr, "Sklearn tree step: %d, mean squared error: %f" % (i, mean_squared_error(current_model_predictions, y))
            # print >>sys.stderr, current_model_predictions

        self.models = models
        self.coefficients = coefficients

    def predict(self, x):
        """
        :param x: 2D array of features x[object][feature]
        :return:
        """
        x = np.array(x)
        predictions = np.zeros(x.shape[0])
        for i, tree in enumerate(self.models):
            predictions += self.coefficients[i] * tree.predict(x)

        return predictions