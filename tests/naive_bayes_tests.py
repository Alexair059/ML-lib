import numpy as np

from utils.testing import random_tensor

from supervised_models.naive_bayes import GaussianNBClassifier
from sklearn import naive_bayes

def test_GaussianNBClassifier():
    np.random.seed(42)
    n_ex = np.random.randint(1, 1000)
    n_feats = np.random.randint(1, 100)
    n_classes = np.random.randint(2, 10)

    X = random_tensor((n_ex, n_feats), standardize=True)
    y = np.random.randint(0, n_classes, size=n_ex)

    X_test = random_tensor((n_ex, n_feats), standardize=True)

    NB = GaussianNBClassifier(eps=1e-09)
    NB.fit(X, y)
    pred = NB.predict(X_test)

    sklearn_NB = naive_bayes.GaussianNB()
    sklearn_NB.fit(X, y)
    sk_pred = sklearn_NB.predict(X_test)

    matching_rate = sum(pred == sk_pred) / len(pred)

    print("Data examples: ", n_ex)
    print("Data features: ", n_feats)
    print("Data classes: ", n_classes)
    print("Prediction matching rate between your NBC and sklearn:", matching_rate)