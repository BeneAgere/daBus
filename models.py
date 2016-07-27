from sklearn.cross_validation import KFold, train_test_split
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 75)

# metrics
def rmsle(theta, thetahat):
    ''' Compute Root-mean-squared-log-error '''
    return np.sqrt(np.mean(np.log(np.array(theta)+1) - np.log(np.array(thetahat)+1)) ** 2)

# make custom score function
custom_score = make_scorer(rmsle, greater_is_better=True)

# base model
def eval_base_model(estimator, xtrain, xtest, ytrain, ytest):
        ''' Calculate linear regression on the features to predict target'''
        estimator  = estimator.fit(xtrain, ytrain)
        yhat_train = estimator.predict(xtrain)
        yhat_test  = estimator.predict(xtest)

        err_train  = rmsle(ytrain, yhat_train)
        err_test   = rmsle(ytest,  yhat_test)
        return {"Model Name": (estimator.__class__.__name__),
                "Err Train": err_train, "Err Test": err_test}

# cross validation
def kfolds_cv(estimator, X, y):
    num_folds = 10
    kf = KFold(len(X), n_folds=num_folds, shuffle=True)

    yhat_train = np.zeros(len(y), dtype = y.dtype)
    yhat_test  = np.zeros(len(y), dtype = y.dtype)
    train_err  = []
    test_err   = []

    for train_idx, test_idx in kf:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Scale the data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        # fit the estimator (estimator.__class__.__name__)
        estimator  = estimator.fit(X_train_scaled, y_train)
        yhat_train = estimator.predict(X_train_scaled)
        yhat_test  = estimator.predict(X_test_scaled)
        # store train and test error
        train_err.append( rmsle(y_train, yhat_train) )
        test_err.append(  rmsle(y_test,  yhat_test) )

    return {"Model Name":(estimator.__class__.__name__),
            "Err Train": np.mean(train_err),
            "Err Test": np.mean(test_err)}

# grid search
def grid_search_cv():
    lg_base_params = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    ridge_params   = {'alpha':[0.001, 0.01,0.1,0.2,0.3,0.4,0.5], 'normalize':[True,False]}
    lasso_params   = {'alpha':[0.001, 0.01,0.1,0.2,0.3,0.4,0.5], 'normalize':[True,False]}

    estimators = [(LinearRegression, lg_base_params),
                  (Ridge, ridge_params),
                  (Lasso, lasso_params)]

    for est, params in estimators:
        gridsearch = GridSearchCV(est(),
                                  params,
                                  n_jobs=-1,
                                  verbose=True,
                                  scoring=custom_score)
        gridsearch.fit(xtrain, ytrain)
        print 'estimator name:', gridsearch.best_estimator_
        print "best score:", gridsearch.best_score_
        print "best parameters:", gridsearch.best_params_
        best_model = gridsearch.best_estimator_


if __name__ == "__main__":
    df_train = pd.read_csv("data/train.csv", parse_dates=['saledate'], low_memory=False)
    df_test = pd.read_csv("data/test.csv", parse_dates=['saledate'], low_memory=False)
    # replace with proper feature set
    df_sub = df_train[['YearMade', 'MachineHoursCurrentMeter', 'SalePrice']].dropna()

    features =  df_sub.drop('SalePrice', axis=1)
    target   =  df_sub['SalePrice']
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.3)

    # perform grid search
    grid_search_cv()

    models = [LinearRegression(normalize=True),
          Ridge(alpha=0.2, normalize=True, solver='auto'),
          Lasso(alpha=0.2, normalize=True),
          RandomForestRegressor(n_jobs=-1),
          GradientBoostingRegressor()]

    for model in models:
        print eval_base_model(model, xtrain, xtest, ytrain, ytest)
        print kfolds_cv(model, features.values, target.values)
