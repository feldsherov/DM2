import unittest
import sklearn.metrics
import numpy as np

from DecisionTreeRegressor import DecisionTreeRegressor


class DecisionTreeTest(unittest.TestCase):
    TestDataPath = "../DataSets/test.data.txt"

    def test_on_test_dataset(self):
        tree = DecisionTreeRegressor()
        data = np.loadtxt(DecisionTreeTest.TestDataPath)
        tree.fit(data[::, :-1:], data[::, -1])

if __name__ == "__main__":
    unittest.main()