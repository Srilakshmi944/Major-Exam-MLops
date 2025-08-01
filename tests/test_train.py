import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import joblib
from sklearn.linear_model import LinearRegression
import utils

def test_data_loading():
    X_train, X_test, y_train, y_test = utils.load_data()
    assert X_train.shape[0] > 0

def test_model_training():
    X_train, _, y_train, _ = utils.load_data()
    model = utils.train_model(X_train, y_train)
    assert isinstance(model, LinearRegression)
    assert hasattr(model, 'coef_')

def test_model_accuracy():
    X_train, X_test, y_train, y_test = utils.load_data()
    model = utils.train_model(X_train, y_train)
    _, r2 = utils.evaluate_model(model, X_test, y_test)
    assert r2 > 0.5