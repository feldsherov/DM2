import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from sklearn.tree import  DecisionTreeRegressor as SklearnDecisionTreeRegressor

from DecisionTreeRegressor import DecisionTreeRegressor
__author__ = 'feldsherov'


class GradientTreeBoosting:
    def __init__(self, count_steps=200, b_coef=1e-2, max_tree_depth=8):
        self.count_steps = count_steps
        self.b_coef = b_coef
        self.max_tree_depth = max_tree_depth
        self.coefficients = None
        self.models = None

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)

        models, coefficients = list(), list()
        models.append(DecisionTreeRegressor(max_depth=self.max_tree_depth))
        models[-1].fit(x, y)
        coefficients.append(1e-3)

        current_model_predictions = coefficients[-1] * models[-1].predict(x)

        for i in range(self.count_steps):
            antigrad = 2*(y - current_model_predictions)
            models.append(DecisionTreeRegressor(max_depth=self.max_tree_depth))
            models[-1].fit(x, antigrad)
            coefficients.append(self.b_coef)

            current_model_predictions += coefficients[-1] * models[-1].predict(x)

            #print >>sys.stderr, antigrad
            print >>sys.stderr, mean_squared_error(current_model_predictions, y), i

        self.models = models
        self.coefficients = coefficients


    def predict(self, x):
        x = np.array(x)
        predictions = np.zeros(x.shape[0])
        for i, tree in enumerate(self.models):
            predictions += self.coefficients[i] * tree.predict(x)

        return predictions


class GradientTreeBoostingViaSklearnTree:
    def __init__(self, count_steps=400, b_coef=1e-3, max_tree_depth=10, min_branch_size=0.2):
        self.count_steps = count_steps
        self.b_coef = b_coef
        self.max_tree_depth = max_tree_depth
        self.min_branch_size = min_branch_size
        self.coefficients = None
        self.models = None

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)


        models, coefficients = list(), list()
        models.append(SklearnDecisionTreeRegressor(max_depth=self.max_tree_depth))
        models[-1].fit(x, y)
        coefficients.append(1e-3)

        current_model_predictions = coefficients[-1] * models[-1].predict(x)

        for i in range(self.count_steps):
            antigrad = 2*(y - current_model_predictions)
            models.append(SklearnDecisionTreeRegressor(max_depth=self.max_tree_depth))
            models[-1].fit(x, antigrad)
            coefficients.append(self.b_coef)

            current_model_predictions += coefficients[-1] * models[-1].predict(x)

            #print >>sys.stderr, mean_squared_error(current_model_predictions, y)

        self.models = models
        self.coefficients = coefficients

    def predict(self, x):
        x = np.array(x)
        predictions = np.zeros(x.shape[0])
        for i, tree in enumerate(self.models):
            predictions += self.coefficients[i] * tree.predict(x)

        return predictions