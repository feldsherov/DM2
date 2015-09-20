import unittest
import logging
import sys
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn import tree as sklearn_trees

from DecisionTreeRegressor import DecisionTreeRegressor

__author__ = 'feldsherov'


class DecisionTreeTest(unittest.TestCase):
    TestDataPath = "../DataSets/test.data.txt"
    HousingDataPath = "../DataSets/housing.data.txt"
    AutoDataPath = "../DataSets/auto-mpg.data.txt"
    SPAMDataPath = "../DataSets/spam.train.txt"

    def test_on_test_dataset(self):
        log = logging.getLogger("DecisionTreeTest.test_on_test_dataset")
        tree = DecisionTreeRegressor()
        data = np.loadtxt(DecisionTreeTest.TestDataPath)
        tree.fit(data[::, :-1:], data[::, -1])

        prediction = tree.predict(data[::, :-1:])
        y = data[::, -1]

        log.debug("Prediction: {0}".format(prediction))
        log.debug("Target value: {0}".format(y))

        self.assertTrue(np.array_equal(prediction, y))

    def test_on_housing_dataset(self):
        log = logging.getLogger("DecisionTreeTest.test_on_housing_dataset")
        data = np.loadtxt(DecisionTreeTest.HousingDataPath)

        x, y = data[::, :-1:], data[::, -1]

        kf = KFold(x.shape[0], n_folds=5)

        for train, test in kf:
            train_x, train_y = x[train], y[train]
            test_x, test_y = x[test], y[test]

            tree = DecisionTreeRegressor(max_depth=50, min_list_size=2, min_list_variance=1e-5, min_branch_size=0.2)
            tree.fit(train_x, train_y)

            sktree = sklearn_trees.DecisionTreeRegressor()
            sktree.fit(train_x, train_y)

            prediction = tree.predict(test_x)
            skprediction = sktree.predict(test_x)

            log.debug("Target: %s" % test_y)
            log.debug("Prediction: %s" % prediction)
            log.debug("Mean squared error my tree: %f" % mean_squared_error(test_y, prediction))
            log.debug("Mean squared error sklearn tree: %f" % mean_squared_error(test_y, skprediction))




if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("DecisionTreeTest.test_on_test_dataset").setLevel(logging.DEBUG)
    logging.getLogger("DecisionTreeTest.test_on_housing_dataset").setLevel(logging.DEBUG)
    logging.getLogger("DecisionTreeTest.test_on_spam_dataset").setLevel(logging.DEBUG)
    unittest.main()