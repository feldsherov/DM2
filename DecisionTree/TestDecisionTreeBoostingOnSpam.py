import sys
import logging
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn import tree as sklearn_trees
from sklearn.ensemble import GradientBoostingRegressor

from GradientTreeBoosting import GradientTreeBoosting, GradientTreeBoostingViaSklearnTree


def main():
    SpamTrain = "../DataSets/spam.train.txt"
    SpamTest = "../DataSets/spam.test.txt"

    logging.basicConfig(stream=sys.stderr)
    log = logging.getLogger("GradientTreeBoosting.test_on_spam_dataset")
    log.setLevel(logging.DEBUG)

    data = np.loadtxt(SpamTrain)

    train_x, train_y = data[::, 1::], data[::, 0]

    ensemble = GradientTreeBoosting(count_steps=200, max_tree_depth=3, step=1e-2, debug=True)
    ensemble.fit(train_x, train_y)

    ensemble_with_sktree = GradientTreeBoostingViaSklearnTree(count_steps=200, max_tree_depth=3, step=1e-2, debug=True)
    ensemble_with_sktree.fit(train_x, train_y)

    # for tree in ensemble.models:
    #     tree.visualize()
    #     print >>sys.stderr, "\n"
    #
    # for tree in ensemble_with_sktree.models:
    #     sklearn_trees.export_graphviz(tree, out_file=sys.stderr)
    #     print >>sys.stderr, "\n"


    sktree = sklearn_trees.DecisionTreeRegressor()
    sktree.fit(train_x, train_y)

    skensemble = GradientBoostingRegressor(n_estimators=200, max_depth=3)
    skensemble.fit(train_x, train_y)


    data = np.loadtxt(SpamTest)

    test_x, test_y = data[::, 1::], data[::, 0]

    prediction = ensemble.predict(test_x)
    skprediction = sktree.predict(test_x)
    skboosting_prediction = skensemble.predict(test_x)
    boosting_with_sktree_prediction = ensemble_with_sktree.predict(test_x)

    log.debug("Target: %s" % test_y)
    log.debug("Prediction my boosting: %s" % prediction)
    log.debug("Prediction boosting with sklearn tree: %s" % boosting_with_sktree_prediction)

    log.debug("Mean squared error my boosting: %f" % mean_squared_error(test_y, prediction))
    log.debug("Mean squared error sklearn tree: %f" % mean_squared_error(test_y, skprediction))


    log.debug("Mean squared error boosting with sklearn tree: %f" %
             mean_squared_error(test_y, boosting_with_sktree_prediction))
    log.debug("Mean squared error sklearn boosting: %f" % mean_squared_error(test_y, skboosting_prediction))


if __name__ == "__main__":
    main()
