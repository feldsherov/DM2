import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def l2dist(pred, y):
    return np.sum((pred - y)**2, axis=1).reshape(pred.shape[0], 1)


def d_l2dist(pred, y):
    return 2 * (pred - y)


def softmax(z):
    ex = np.exp(z)
    return ex / np.sum(ex, axis=1).reshape(ex.shape[0], 1)


def d_softmax(m):
    y = softmax(m)
    return y * (1 - y)


class MLP:
    def __init__(self, input_dimension=None,
                 layers=None,
                 f_activation=None,
                 df_activation=None,
                 n_steps=200,
                 seed=179,
                 regularization='L2',
                 learning_rate=0.01):
        np.random.seed(seed)

        if input_dimension is None:
            raise ValueError("Input dimension must not be None")

        if layers is None:
            self.n_layers = 10
            layers = [self.n_layers for _ in range(self.n_layers)]
        self.n_layers = len(layers)

        if f_activation is None or df_activation is None:
            f_activation = [sigmoid for _ in range(self.n_layers)]
            df_activation = [d_sigmoid for _ in range(self.n_layers)]

        self.w = [None] * self.n_layers
        for i in range(0, self.n_layers):
            prev = (layers[i - 1] if i != 0 else input_dimension)
            nxt = layers[i]
            self.w[i] = np.random.uniform(-0.1, 0.1, (prev + 1, nxt))

        self.layers = layers
        self.f_activation = f_activation
        self.df_activation = df_activation

        self.loss = l2dist
        self.d_loss = d_l2dist
        self.n_steps = n_steps
        self.regularization = regularization

        self.learning_rate = learning_rate

    def fit(self, X, Y, val_x, val_y, add_bias=True):
        if add_bias:
            X = np.concatenate((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)), axis=1)

        err_tr = list()
        err_val = list()
        Xs = [None] * self.n_layers
        fXs = [None] * self.n_layers

        for t in range(self.n_steps):
            tmp = X
            for i in range(self.n_layers):
                Xs[i] = np.dot(tmp, self.w[i])
                tmp = self.f_activation[i](Xs[i])
                fXs[i] = tmp
                if i != self.n_layers - 1:
                    tmp = np.concatenate((tmp, np.ones(tmp.shape[0]).reshape(tmp.shape[0], 1)), axis=1)

            err_tr.append(self.loss(fXs[-1], Y).mean())
            err_val.append(self.loss(self.predict(val_x), val_y).mean())

            gradW = [None] * self.n_layers
            dE_n = None
            for i in reversed(range(self.n_layers)):
                dE = None
                if i == self.n_layers - 1:
                    dE = self.d_loss(fXs[i], Y) * self.df_activation[i](Xs[i])
                    dE_n = dE
                else:
                    dE = np.dot(self.w[i + 1][:-1, :], dE_n.T).T * self.df_activation[i](Xs[i])
                    dE_n = dE

                inp = None
                if i == 0:
                    inp = X
                else:
                    inp = np.concatenate((fXs[i - 1], np.ones(fXs[i - 1].shape[0]).reshape(fXs[i - 1].shape[0], 1)), axis=1)

                gradW[i] = np.dot(inp.T, dE) / len(X)

            #print gradW
            for i in range(self.n_layers):
                if self.regularization == 'L2':
                    gradW[i] += 2*self.w[i]
                self.w[i] -= self.learning_rate * gradW[i]

        return err_tr, err_val

    def predict(self, X, add_bias=True):
        if add_bias:
            X = np.concatenate((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)), axis=1)

        tmp = np.copy(X)
        #print tmp
        for i in range(0, self.n_layers):
            tmp = np.dot(tmp, self.w[i])
            tmp = self.f_activation[i](tmp)
            #print tmp
            if i != self.n_layers - 1:
                tmp = np.concatenate((tmp, np.ones(tmp.shape[0]).reshape(tmp.shape[0], 1)), axis=1)

        return tmp

if __name__ == "__main__":
    mlp = MLP(input_dimension=3,
              layers=([3, 3]),
              f_activation=[sigmoid, softmax],
              df_activation=[d_sigmoid, d_softmax],
              seed=179,
              n_steps=100)
    x = np.array([[1, 2, 1] for i in range(1, 100)])
    y = np.array([[0, 0, 1] for i in range(1, 100)])
    err, val_err = mlp.fit(x, y, x, y)
    print err
    print mlp.predict(x)
