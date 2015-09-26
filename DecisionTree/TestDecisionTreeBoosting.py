import unittest
import logging
import sys
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn import tree as sklearn_trees
from sklearn.ensemble import GradientBoostingRegressor

from GradientTreeBoosting import GradientTreeBoosting, GradientTreeBoostingViaSklearnTree


__author__ = 'Svyatoslav Feldsherov'


class DecisionTreeTest(unittest.TestCase):
    """
    Unit test class for GradientTreeBoosting testing
    """
    TestDataPath = "../DataSets/test.data.txt"
    HousingDataPath = "../DataSets/housing.data.txt"
    AutoDataPath = "../DataSets/auto-mpg.data.txt"
    SPAMDataPath = "../DataSets/spam.train.txt"

    def test_on_housing_dataset(self):
        """
        Test on housing data set
        Logging result and MSE for some different models
        :return: None
        """
        log = logging.getLogger("GradientTreeBoosting.test_on_housing_dataset")
        data = np.loadtxt(DecisionTreeTest.HousingDataPath)

        x, y = data[::, :-1:], data[::, -1]

        kf = KFold(x.shape[0], n_folds=5)

        for train, test in kf:
            train_x, train_y = x[train], y[train]
            test_x, test_y = x[test], y[test]

            ensemble = GradientTreeBoosting(count_steps=200, step=1e-2, max_tree_depth=8)
            ensemble.fit(train_x, train_y)

            ensemble_with_sktree = GradientTreeBoostingViaSklearnTree(count_steps=200, max_tree_depth=8, step=1e-2)
            ensemble_with_sktree.fit(train_x, train_y)

            sktree = sklearn_trees.DecisionTreeRegressor()
            sktree.fit(train_x, train_y)

            skensemble = GradientBoostingRegressor()
            skensemble.fit(train_x, train_y)

            prediction = ensemble.predict(test_x)
            skprediction = sktree.predict(test_x)
            skboosting_prediction = skensemble.predict(test_x)
            boosting_with_sktree_prediction = ensemble_with_sktree.predict(test_x)

            log.debug("Target: %s" % test_y)
            log.debug("Prediction: %s" % prediction)
            log.debug("Mean squared error my boosting: %f" % mean_squared_error(test_y, prediction))
            log.debug("Mean squared error sklearn tree: %f" % mean_squared_error(test_y, skprediction))
            log.debug("Mean squared error boosting with sklearn tree: %f" %
                      mean_squared_error(test_y, boosting_with_sktree_prediction))
            log.debug("Mean squared error sklearn boosting: %f" % mean_squared_error(test_y, skboosting_prediction))


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("GradientTreeBoosting.test_on_housing_dataset").setLevel(logging.DEBUG)
    unittest.main()