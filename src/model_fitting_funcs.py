import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import time
from scipy.stats import zscore
from cvxopt import matrix, solvers
import torch
import gc
import torch.nn as nn


# Turn off torch gradient tracking
torch.set_grad_enabled(False)

# Set option to not show progress in CVXOPT solver
solvers.options["show_progress"] = False


def r2_multi(y_true, y_pred):
    SSres = np.mean((y_true - y_pred) ** 2, 0)
    SStot = np.var(y_true, 0)
    r2 = np.nan_to_num(1 - SSres / SStot)
    return r2

def corr_multi(y_true, y_pred):
    # centering
    y_true_centered = y_true - np.mean(y_true, axis=0)
    y_pred_centered = y_pred - np.mean(y_pred, axis=0)

    # calculate column-wise correlation
    numerator = np.sum(y_true_centered * y_pred_centered, axis=0)
    denominator = np.sqrt(np.sum(y_true_centered**2, axis=0) * np.sum(y_pred_centered**2, axis=0))
    corrs = numerator / denominator
    return corrs

def ridge_alphas_err(trn_X, trn_Y, val_X, val_Y, alphas):
    # if data too large, we calculate each alpha separately
    if (trn_X.size + trn_Y.size) * len(alphas) > 1e9:
        errs = np.zeros((len(alphas), trn_Y.shape[1]))
        for i, alpha in enumerate(alphas):
            ridge = Ridge(alpha=alpha, fit_intercept=False)
            ridge.fit(trn_X, trn_Y)
            val_Y_pred = ridge.predict(val_X)
            errs[i, :] = 1 - r2_multi(val_Y, val_Y_pred)

    # otherwise, we can calculate all alphas at once
    else:
        alphas_flattened = np.repeat(alphas, trn_Y.shape[1])
        trn_Y_expanded = np.tile(trn_Y, (1, len(alphas)))
        val_Y_expanded = np.tile(val_Y, (1, len(alphas)))
        ridge = Ridge(alpha=alphas_flattened, fit_intercept=False)
        ridge.fit(trn_X, trn_Y_expanded)
        val_Y_pred_expanded = ridge.predict(val_X)
        errs_expanded = 1 - r2_multi(val_Y_expanded, val_Y_pred_expanded)
        errs = errs_expanded.reshape((len(alphas), trn_Y.shape[1]))
    
    return errs


def cv_ridge(train_features, train_targets, alphas=[10**j for j in range(-6, 10)], n_splits=10, print_time=True):
    
    # split the data into training and validation and calculate errors for each alpha and each channel
    kfold = KFold(n_splits=n_splits)
    errs_cv = np.zeros((len(alphas), train_targets.shape[1]))
    for trn_idx, val_idx in kfold.split(train_features):
        start_time = time.time()
        trn_X, trn_Y = train_features[trn_idx], train_targets[trn_idx]
        val_X, val_Y = train_features[val_idx], train_targets[val_idx]
        errs = ridge_alphas_err(trn_X, trn_Y, val_X, val_Y, alphas)
        errs_cv += errs
        if print_time:
            print(f"Time taken for one fold: {time.time() - start_time}")
    
    # get the best alpha for each channel
    best_alpha_idx = np.argmin(errs_cv, axis=0)
    best_alphas = np.array([alphas[i] for i in best_alpha_idx])
    
    return best_alphas


def ridge_fit_predict(train_features, train_targets, test_features, test_targets, alphas=[10**j for j in range(-6, 10)], n_splits=10, return_coefs=False):
    
    # get the best alphas
    best_alphas = cv_ridge(train_features, train_targets, alphas=alphas, n_splits=n_splits)
    
    # fit the ridge regression model
    ridge = Ridge(alpha=best_alphas, fit_intercept=False)
    ridge.fit(train_features, train_targets)
    test_targets_pred = ridge.predict(test_features)
    
    # get the correlation and r2 for each channel
    corrs = corr_multi(test_targets, test_targets_pred)
    r2s = r2_multi(test_targets, test_targets_pred)
    
    if return_coefs:
        return corrs, r2s, ridge
    else:
        return corrs, r2s


def ridge_for_stacking(train_features, train_targets, test_features, alphas=[10**j for j in range(-6, 10)], n_splits=5, score_function=r2_multi, return_coefs=False):

    ## predict train, need to manually split (train, test) within train, cv alpha + fit ridge for each split
    start_time = time.time()
    kfold = KFold(n_splits=n_splits)
    train_preds = np.zeros_like(train_targets)
    for train_trn_idx, train_val_idx in kfold.split(train_features):
        train_trn_X, train_trn_Y = train_features[train_trn_idx], train_targets[train_trn_idx]
        train_val_X = train_features[train_val_idx]
        train_best_alphas = cv_ridge(train_trn_X, train_trn_Y, alphas=alphas, n_splits=n_splits, print_time=False)
        ridge = Ridge(alpha=train_best_alphas, fit_intercept=False)
        ridge.fit(train_trn_X, train_trn_Y)
        train_val_Y_pred = ridge.predict(train_val_X)
        train_preds[train_val_idx] = train_val_Y_pred
    print(f"Time taken for predicting train: {time.time() - start_time}")

    ## then for all train, cv alpha + fit ridge
    # get the best alphas
    best_alphas = cv_ridge(train_features, train_targets, alphas=alphas, n_splits=n_splits)

    # fit the ridge regression model
    ridge = Ridge(alpha=best_alphas, fit_intercept=False)
    ridge.fit(train_features, train_targets)
    test_preds = ridge.predict(test_features)

    ## get the score on the training set
    train_err = train_targets - train_preds
    train_scores = score_function(train_targets, train_preds)
    train_variances = np.var(train_preds, axis=0)

    if return_coefs:
        return train_preds, train_err, test_preds, train_scores, train_variances, ridge
    else:
        return train_preds, train_err, test_preds, train_scores, train_variances


def stacking(train_data, test_data, train_features, test_features, score_f=r2_multi, alphas=[10**j for j in range(-6, 10)], return_ridges=False):
    """
    Stacks predictions from different feature spaces and uses them to make final predictions.

    Args:
        train_data (ndarray): Training data of shape (n_time_train, n_voxels)
        test_data (ndarray): Testing data of shape (n_time_test, n_voxels)
        train_features (list): List of training feature spaces, each of shape (n_time_train, n_dims)
        test_features (list): List of testing feature spaces, each of shape (n_time_test, n_dims)
        score_f (callable): Scikit-learn scoring function to use for evaluation. Default is mean_squared_error.

    Returns:
        Tuple of ndarrays:
            - r2s: Array of shape (n_features, n_voxels) containing unweighted R2 scores for each feature space and voxel
            - stacked_r2s: Array of shape (n_voxels,) containing R2 scores for the stacked predictions of each voxel
            - r2s_weighted: Array of shape (n_features, n_voxels) containing R2 scores for each feature space weighted by stacking weights
            - r2s_train: Array of shape (n_features, n_voxels) containing R2 scores for each feature space and voxel in the training set
            - stacked_train_r2s: Array of shape (n_voxels,) containing R2 scores for the stacked predictions of each voxel in the training set
            - S: Array of shape (n_voxels, n_features) containing the stacking weights for each voxel
    """

    # [bj] Note: r2_multi argument order is reversed than R2!
    # Number of time points in the test set
    n_time_test = test_data.shape[0]

    # Check that the number of voxels is the same in the training and test sets
    assert train_data.shape[1] == test_data.shape[1]
    n_voxels = train_data.shape[1]

    # Check that the number of feature spaces is the same in the training and test sets
    assert len(train_features) == len(test_features)
    n_features = len(train_features)

    # Array to store R2 scores for each feature space and voxel
    r2s = np.zeros((n_features, n_voxels))
    # Array to store R2 scores for each feature space and voxel in the training set
    r2s_train = np.zeros((n_features, n_voxels))
    # Array to store variance explained by the model for each feature space and voxel in the training set
    var_train = np.zeros((n_features, n_voxels))
    # Array to store R2 scores for each feature space weighted by stacking weights
    r2s_weighted = np.zeros((n_features, n_voxels))

    # Array to store stacked predictions for each voxel
    stacked_pred = np.zeros((n_time_test, n_voxels))
    # Dictionary to store predictions for each feature space and voxel in the training set
    preds_train = {}
    # Dictionary to store predictions for each feature space and voxel in the test set
    preds_test = np.zeros((n_features, n_time_test, n_voxels))
    # Array to store weighted predictions for each feature space and voxel in the test set
    weighted_pred = np.zeros((n_features, n_time_test, n_voxels))

    # normalize data by TRAIN/TEST
    train_data = np.nan_to_num(zscore(train_data))
    test_data = np.nan_to_num(zscore(test_data))

    train_features = [np.nan_to_num(zscore(F)) for F in train_features]
    test_features = [np.nan_to_num(zscore(F)) for F in test_features]

    # initialize an error dictionary to store errors for each feature
    err = dict()
    preds_train = dict()

    # iterate over each feature and train a model using feature ridge regression
    ridges = []
    for FEATURE in range(n_features):
        (
            preds_train[FEATURE],
            error,
            preds_test[FEATURE, :, :],
            r2s_train[FEATURE, :],
            var_train[FEATURE, :],
            ridge,
        ) = ridge_for_stacking(
            train_features[FEATURE],
            train_data,
            test_features[FEATURE],
            alphas=alphas,
            n_splits=5,
            return_coefs=True,
        )
        err[FEATURE] = error
        ridges.append(ridge)

    # calculate error matrix for stacking
    P = np.zeros((n_voxels, n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            P[:, i, j] = np.mean(err[i] * err[j], 0)

    # solve the quadratic programming problem to obtain the weights for stacking
    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    S = np.zeros((n_voxels, n_features))
    stacked_pred_train = np.zeros_like(train_data)

    for i in range(0, n_voxels):
        PP = matrix(P[i])
        # solve for stacking weights for every voxel
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b, show_progress=False)["x"]).reshape(n_features)

        # combine the predictions from the individual feature spaces for voxel i
        z_test = np.array([preds_test[feature_j, :, i] for feature_j in range(n_features)])
        z_train = np.array([preds_train[feature_j][:, i] for feature_j in range(n_features)])
        # multiply the predictions by S[i,:]
        stacked_pred[:, i] = np.dot(S[i, :], z_test)
        # combine the training predictions from the individual feature spaces for voxel i
        stacked_pred_train[:, i] = np.dot(S[i, :], z_train)

    # compute the R2 score for the stacked predictions on the training data
    stacked_train_r2s = score_f(train_data, stacked_pred_train)

    # compute the R2 scores for each individual feature and the weighted feature predictions
    for FEATURE in range(n_features):
        # weight the predictions according to S:
        # weighted single feature space predictions, computed over a fold
        weighted_pred[FEATURE, :] = preds_test[FEATURE, :] * S[:, FEATURE]

    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(test_data, preds_test[FEATURE])
        r2s_weighted[FEATURE, :] = score_f(test_data, weighted_pred[FEATURE])

    # compute the R2 score for the stacked predictions on the test data
    stacked_r2s = score_f(test_data, stacked_pred)

    # return the results
    if return_ridges:
        return r2s, stacked_r2s, r2s_weighted, r2s_train, stacked_train_r2s, S, ridges
    else:
        return r2s, stacked_r2s, r2s_weighted, r2s_train, stacked_train_r2s, S


## -------- Following are for GPU -------- ##

def zscore_tensor(tensor, dim=0):
    """Z-score a tensor along a given dimension"""
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True, correction=0)
    tensor_norm = torch.where(std > 0, (tensor - mean) / std, torch.zeros_like(tensor))
    return tensor_norm


def r2_multi_torch(y_true, y_pred):
    """
    Compute R² score for each target dimension using PyTorch tensors.
    Inputs: (n_samples, n_targets)
    """
    ss_res = torch.mean((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.var(y_true, dim=0, unbiased=False)

    r2 = 1 - ss_res / ss_tot
    r2[torch.isnan(r2)] = 0
    r2[torch.isinf(r2)] = 0
    return r2


def corr_multi_torch(y_true, y_pred, dim=0):
    """
    Compute correlation between y_true and y_pred per target dimension.
    Inputs: (n_samples, n_targets)
    """
    # y_true_z = (y_true - y_true.mean(dim=dim, keepdim=True)) / (y_true.std(dim=dim, keepdim=True) + 1e-8)
    # y_pred_z = (y_pred - y_pred.mean(dim=dim, keepdim=True)) / (y_pred.std(dim=dim, keepdim=True) + 1e-8)
    y_true_z = zscore_tensor(y_true, dim=dim)
    y_pred_z = zscore_tensor(y_pred, dim=dim)
    
    return torch.mean(y_true_z * y_pred_z, dim=dim)


def ridge_torch(X, Y, alpha):
    """
    Ridge regression using PyTorch on GPU.
    
    Parameters:
    - X: (n_samples, n_features) torch.Tensor
    - Y: (n_samples,) or (n_samples, n_targets) torch.Tensor
    - alpha: float, regularization strength
    
    Returns:
    - weights: (n_features,) or (n_features, n_targets) torch.Tensor
    """
    device = X.device
    n_features = X.shape[1]
    XtX = X.T @ X
    ridge_term = alpha * torch.eye(n_features, device=device, dtype=X.dtype)
    XtY = X.T @ Y
    weights = torch.linalg.solve(XtX + ridge_term, XtY)
    return weights


def ridge_alphas_err_torch(trn_X, trn_Y, val_X, val_Y, alphas):
    """
    Compute validation error for multiple ridge alphas using PyTorch tensors.
    All inputs are expected to be torch tensors already on the right device.
    """
    errs = torch.zeros((len(alphas), trn_Y.shape[1]), device=trn_X.device)

    for i, alpha in enumerate(alphas):
        weights = ridge_torch(trn_X, trn_Y, alpha)
        val_Y_pred = val_X @ weights
        errs[i, :] = 1 - r2_multi_torch(val_Y, val_Y_pred)
        del val_Y_pred, weights

    return errs.cpu().numpy()


def cv_ridge_torch(train_X, train_Y, alphas=[10**j for j in range(-6, 10)], n_splits=10, print_time=True):
    """
    Cross-validated ridge regression to select best alpha per target dimension.
    Inputs are torch tensors.
    """
    errs_cv = np.zeros((len(alphas), train_Y.shape[1]))
    
    kfold = KFold(n_splits=n_splits)
    train_X_np = train_X.cpu().numpy()  # kfold split only work on numpy array

    for trn_idx, val_idx in kfold.split(train_X_np):
        start_time = time.time()

        trn_X = train_X[trn_idx]
        trn_Y = train_Y[trn_idx]
        val_X = train_X[val_idx]
        val_Y = train_Y[val_idx]

        errs = ridge_alphas_err_torch(trn_X, trn_Y, val_X, val_Y, alphas)
        errs_cv += errs

        if print_time:
            print(f"Time taken for one fold: {time.time() - start_time:.2f} sec")
        
        del trn_X, trn_Y, val_X, val_Y  # save GPU memory

    best_alpha_idx = np.argmin(errs_cv, axis=0)
    best_alphas = np.array([alphas[i] for i in best_alpha_idx])
    return best_alphas


def _fit_ridge_per_target_alpha(X, Y, best_alphas):
    """
    Fit ridge regression for each target dimension using the best alphas.
    """
    device = X.device
    n_feat = X.shape[1]
    n_targets = Y.shape[1]
    W = torch.zeros((n_feat, n_targets), device=device, dtype=X.dtype)

    alpha_to_cols = {}
    for col, a in enumerate(best_alphas):
        alpha_to_cols.setdefault(a, []).append(col)

    for alpha, cols in alpha_to_cols.items():
        cols_t = torch.tensor(cols, device=device)
        W[:, cols_t] = ridge_torch(X, Y[:, cols_t], alpha)

    return W


def ridge_fit_predict_torch(train_features, train_targets, test_features, test_targets,
                            alphas=[10**j for j in range(-6, 10)], n_splits=10, return_coefs=False, device='cuda'):

    # Convert all inputs to PyTorch tensors on the correct device
    train_X = torch.tensor(train_features, dtype=torch.float32, device=device)
    train_Y = torch.tensor(train_targets, dtype=torch.float32, device=device)
    test_X = torch.tensor(test_features, dtype=torch.float32, device=device)
    test_Y = torch.tensor(test_targets, dtype=torch.float32, device=device)

    # Select best alphas via cross-validation
    best_alphas = cv_ridge_torch(train_X, train_Y, alphas=alphas, n_splits=n_splits)

    # Fit ridge models
    weights = _fit_ridge_per_target_alpha(train_X, train_Y, best_alphas)
    weights_cpu = weights.cpu().numpy()

    # Predict and compute metrics
    test_Y_pred = test_X @ weights
    corrs = corr_multi_torch(test_Y, test_Y_pred).cpu().numpy()
    r2s = r2_multi_torch(test_Y, test_Y_pred).cpu().numpy()
    
    # save GPU memory
    del train_X, train_Y, test_X, test_Y, test_Y_pred, weights
    torch.cuda.empty_cache()

    if return_coefs:
        return corrs, r2s, weights_cpu
    else:
        return corrs, r2s


def ridge_for_stacking_torch(train_features, train_targets, test_features, alphas=[10**j for j in range(-6, 10)], n_splits=5, score_function=r2_multi, return_coefs=False, device="cuda"):
    """
    Cross-validated ridge (torch) that also yields out-of-fold predictions.
    """
    # predict train, need to manually split (train, test) within train, cv alpha + fit ridge for each split
    start_time = time.time()
    kfold = KFold(n_splits=n_splits)
    train_preds = np.zeros_like(train_targets)
    for tr_idx, val_idx in kfold.split(train_features):
        train_trn_X = torch.tensor(train_features[tr_idx], dtype=torch.float32, device=device)
        train_trn_Y = torch.tensor(train_targets[tr_idx], dtype=torch.float32, device=device)
        train_val_X = torch.tensor(train_features[val_idx], dtype=torch.float32, device=device)
        train_best_alphas = cv_ridge_torch(train_trn_X, train_trn_Y, alphas=alphas, n_splits=n_splits, print_time=False)
        W = _fit_ridge_per_target_alpha(train_trn_X, train_trn_Y, train_best_alphas)
        train_preds[val_idx] = (train_val_X @ W).cpu().numpy()
        del train_trn_X, train_trn_Y, train_val_X, W  # save GPU memory
        torch.cuda.empty_cache()
    print(f"Time taken for predicting train: {time.time() - start_time}")
    # print(torch.cuda.memory_summary(device=device))

    # get the score on the training set
    train_err = train_targets - train_preds
    train_scores = score_function(train_targets, train_preds)
    train_variances = np.var(train_preds, axis=0)
    
    # then, for all train, cv alpha + fit ridge
    train_features = torch.tensor(train_features, dtype=torch.float32, device=device)
    train_targets = torch.tensor(train_targets, dtype=torch.float32, device=device)
    test_features = torch.tensor(test_features, dtype=torch.float32, device=device)
    best_alphas = cv_ridge_torch(train_features, train_targets, alphas=alphas, n_splits=n_splits)
    weights = _fit_ridge_per_target_alpha(train_features, train_targets, best_alphas)
    test_preds = (test_features @ weights).cpu().numpy()
    weights_cpu = weights.cpu().numpy()
    del train_features, train_targets, test_features, weights  # save GPU memory
    torch.cuda.empty_cache()

    if return_coefs:
        return train_preds, train_err, test_preds, train_scores, train_variances, weights_cpu
    else:
        return train_preds, train_err, test_preds, train_scores, train_variances


def stacking_torch(train_data, test_data, train_features, test_features, score_f=r2_multi, alphas=[10**j for j in range(-6, 10)], return_ridges=False, device="cuda"):
    """
    Mimics the CPU `stacking` routine but executes the heavy ridge parts on GPU.
    """
    # [bj] Note: r2_multi argument order is reversed than R2!
    # Number of time points in the test set
    n_time_test = test_data.shape[0]

    # Check that the number of voxels is the same in the training and test sets
    assert train_data.shape[1] == test_data.shape[1]
    n_voxels = train_data.shape[1]

    # Check that the number of feature spaces is the same in the training and test sets
    assert len(train_features) == len(test_features)
    n_features = len(train_features)

    # Array to store R2 scores for each feature space and voxel
    r2s = np.zeros((n_features, n_voxels))
    # Array to store R2 scores for each feature space and voxel in the training set
    r2s_train = np.zeros((n_features, n_voxels))
    # Array to store variance explained by the model for each feature space and voxel in the training set
    var_train = np.zeros((n_features, n_voxels))
    # Array to store R2 scores for each feature space weighted by stacking weights
    r2s_weighted = np.zeros((n_features, n_voxels))

    # Array to store stacked predictions for each voxel
    stacked_pred = np.zeros((n_time_test, n_voxels))
    # Dictionary to store predictions for each feature space and voxel in the training set
    preds_train = {}
    # Dictionary to store predictions for each feature space and voxel in the test set
    preds_test = np.zeros((n_features, n_time_test, n_voxels))
    # Array to store weighted predictions for each feature space and voxel in the test set
    weighted_pred = np.zeros((n_features, n_time_test, n_voxels))

    # normalize data by TRAIN/TEST
    train_data = np.nan_to_num(zscore(train_data))
    test_data = np.nan_to_num(zscore(test_data))

    train_features = [np.nan_to_num(zscore(F)) for F in train_features]
    test_features = [np.nan_to_num(zscore(F)) for F in test_features]

    # initialize an error dictionary to store errors for each feature
    err = dict()
    preds_train = dict()

    # iterate over each feature and train a model using feature ridge regression
    weights = []
    for FEATURE in range(n_features):
        (
            preds_train[FEATURE],
            error,
            preds_test[FEATURE, :, :],
            r2s_train[FEATURE, :],
            var_train[FEATURE, :],
            W,
        ) = ridge_for_stacking_torch(
            train_features[FEATURE],
            train_data,
            test_features[FEATURE],
            alphas=alphas,
            n_splits=5,
            return_coefs=True,
            device=device,
        )
        err[FEATURE] = error
        weights.append(W)

    # calculate error matrix for stacking
    P = np.zeros((n_voxels, n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            P[:, i, j] = np.mean(err[i] * err[j], 0)

    # solve the quadratic programming problem to obtain the weights for stacking
    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    S = np.zeros((n_voxels, n_features))
    stacked_pred_train = np.zeros_like(train_data)

    for i in range(0, n_voxels):
        PP = matrix(P[i])
        # solve for stacking weights for every voxel
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b, show_progress=False)["x"]).reshape(n_features)

        # combine the predictions from the individual feature spaces for voxel i
        z_test = np.array([preds_test[feature_j, :, i] for feature_j in range(n_features)])
        z_train = np.array([preds_train[feature_j][:, i] for feature_j in range(n_features)])
        # multiply the predictions by S[i,:]
        stacked_pred[:, i] = np.dot(S[i, :], z_test)
        # combine the training predictions from the individual feature spaces for voxel i
        stacked_pred_train[:, i] = np.dot(S[i, :], z_train)

    # compute the R2 score for the stacked predictions on the training data
    stacked_train_r2s = score_f(train_data, stacked_pred_train)

    # compute the R2 scores for each individual feature and the weighted feature predictions
    for FEATURE in range(n_features):
        # weight the predictions according to S:
        # weighted single feature space predictions, computed over a fold
        weighted_pred[FEATURE, :] = preds_test[FEATURE, :] * S[:, FEATURE]

    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(test_data, preds_test[FEATURE])
        r2s_weighted[FEATURE, :] = score_f(test_data, weighted_pred[FEATURE])

    # compute the R2 score for the stacked predictions on the test data
    stacked_r2s = score_f(test_data, stacked_pred)

    # return the results
    if return_ridges:
        return r2s, stacked_r2s, r2s_weighted, r2s_train, stacked_train_r2s, S, weights
    else:
        return r2s, stacked_r2s, r2s_weighted, r2s_train, stacked_train_r2s, S


## -------- Following are for Memory Efficiency -------- ##

def get_lagged_X(X, l, max_L):
    """
    Assume X is padded with max_L 0s in the end. For each row, X0 is the present feature, X1 is the past 1, and so on.
    X_pad = nn.functional.pad(X, pad=(0, 0, 0, max_L))
    """
    assert 0 <= l <= max_L
    X_lagged = torch.roll(X, shifts=l, dims=0)
    return X_lagged


def get_pred_lags(X, W, max_L):
    """Assume X is padded with max_L 0s in the end. W is the flattened weights for all lags."""
    F = X.shape[1]
    preds = sum(get_lagged_X(X, lag, max_L) @ W[lag * F : (lag + 1) * F] for lag in range(max_L + 1))
    return preds


# X: (N, F)  padded feature matrix
# Y: (N, T)  padded targets (can be 1D as (N,) as well)
# L: number of lags (e.g. 30)
# alpha: ridge penalty (float)
def ridge_with_lags(X, Y, max_L, alpha):
    """Assume X and Y are padded with max_L 0s in the end."""
    assert X.shape[0] == Y.shape[0]
    N, F = X.shape
    device, dtype = X.device, X.dtype

    # 1) compute all autocovariance "R_d = A[:N-d].T @ A[d:]" for d=0, ..., max_L
    D = max_L + 1
    Rs = [None] * (D)
    for d in range(D):
        Rs[d] = X[d:].T @ X[: N - d]  # (F, F)

    # 2) build the big (F·D × F·D) Gram matrix block-by-block
    G = torch.zeros((F * D, F * D), device=device, dtype=dtype)
    for i in range(D):
        for j in range(D):
            d = j - i
            if d >= 0:
                block = Rs[d]  # R_{d}
            else:
                block = Rs[-d].T  # R_{-d} = R_d^T
            G[i * F : (i + 1) * F, j * F : (j + 1) * F] = block
    
    # 3) compute X^T Y in D blocks
    XTY = torch.zeros((F * D, Y.shape[1] if Y.ndim > 1 else 1), device=device, dtype=dtype)
    for k in range(D):
        # sum_t A_t^T Y_{t+k} = A[:N-k].T @ Y[k:]
        slice_ = (slice(k, N),) if Y.ndim == 1 else (slice(k, N), slice(None))
        XTY[k * F : (k + 1) * F] = X[: N - k].T @ Y[slice_]

    # 4) solve the normal equations
    ridge_eye = alpha * torch.eye(F * D, device=device, dtype=dtype)
    W_flat = torch.linalg.solve(G + ridge_eye, XTY)
    # reshape back to (D, F, T)
    # return W_flat.view(D, F, -1)
    return W_flat


def ridge_alphas_err_lags(trn_X, trn_Y, val_X, val_Y, max_L, alphas):
    """Assume inputs are padded with max_L 0s in the end."""
    assert trn_X.shape[0] == trn_Y.shape[0]
    assert val_X.shape[0] == val_Y.shape[0]
    assert trn_X.shape[1] == val_X.shape[1]
    assert trn_Y.shape[1] == val_Y.shape[1]
    F = trn_X.shape[1]
    errs = torch.zeros((len(alphas), trn_Y.shape[1]), device=trn_X.device)
    for i, alpha in enumerate(alphas):
        W = ridge_with_lags(trn_X, trn_Y, max_L, alpha)
        preds = get_pred_lags(val_X, W, max_L)
        errs[i] = 1 - r2_multi_torch(val_Y, preds)
        del W, preds
    return errs.cpu().numpy()


def cv_ridge_lags(train_X, train_Y, max_L, alphas=[10**j for j in range(-6, 10)], n_splits=10, print_time=True):
    errs_cv = np.zeros((len(alphas), train_Y.shape[1]))
    kf = KFold(n_splits=n_splits)
    X_np = train_X.cpu().numpy()

    for trn_idx, val_idx in kf.split(X_np):
        start_time = time.time()

        trn_X, trn_Y = train_X[trn_idx], train_Y[trn_idx]
        val_X, val_Y = train_X[val_idx], train_Y[val_idx]
        trn_X = nn.functional.pad(trn_X, pad=(0, 0, 0, max_L))
        trn_Y = nn.functional.pad(trn_Y, pad=(0, 0, 0, max_L))
        val_X = nn.functional.pad(val_X, pad=(0, 0, 0, max_L))
        val_Y = nn.functional.pad(val_Y, pad=(0, 0, 0, max_L))

        errs = ridge_alphas_err_lags(trn_X, trn_Y, val_X, val_Y, max_L, alphas)
        errs_cv += errs
        del trn_X, trn_Y, val_X, val_Y

        if print_time:
            print(f"Time taken for one fold: {time.time() - start_time:.2f} sec")

    best_idx = np.argmin(errs_cv, axis=0)
    return np.array([alphas[i] for i in best_idx])


def _fit_ridge_per_target_alpha_lags(X, Y, max_L, best_alphas):
    """Assume inputs are padded with max_L 0s in the end."""
    F = X.shape[1]
    D = max_L + 1
    n_targets = Y.shape[1]
    W = torch.zeros((F * D, n_targets), device=X.device, dtype=X.dtype)

    alpha_to_cols = {}
    for col, a in enumerate(best_alphas):
        alpha_to_cols.setdefault(a, []).append(col)

    for alpha, cols in alpha_to_cols.items():
        W[:, cols] = ridge_with_lags(X, Y[:, cols], max_L, alpha)

    return W


def ridge_fit_predict_lags(train_features, train_targets, test_features, test_targets, max_L, alphas=[10**j for j in range(-6, 10)], n_splits=10, return_coefs=False, device="cuda"):

    # Select best alphas via cross-validation
    train_X = torch.tensor(train_features, dtype=torch.float32, device=device)
    train_Y = torch.tensor(train_targets, dtype=torch.float32, device=device)
    best_alphas = cv_ridge_lags(train_X, train_Y, max_L, alphas, n_splits)

    # Fit ridge models
    train_X = nn.functional.pad(train_X, pad=(0, 0, 0, max_L))
    train_Y = nn.functional.pad(train_Y, pad=(0, 0, 0, max_L))
    test_X = torch.tensor(test_features, dtype=torch.float32, device=device)
    test_Y = torch.tensor(test_targets, dtype=torch.float32, device=device)
    test_X = nn.functional.pad(test_X, pad=(0, 0, 0, max_L))
    test_Y = nn.functional.pad(test_Y, pad=(0, 0, 0, max_L))
    W = _fit_ridge_per_target_alpha_lags(train_X, train_Y, max_L, best_alphas)
    W_cpu = W.cpu().numpy()

    # Predict and compute metrics
    test_pred = get_pred_lags(test_X, W, max_L)
    corrs = corr_multi_torch(test_Y, test_pred).cpu().numpy()
    r2s = r2_multi_torch(test_Y, test_pred).cpu().numpy()

    # save GPU memory
    del train_X, train_Y, test_X, test_Y, test_pred, W

    if return_coefs:
        return corrs, r2s, W_cpu
    else:
        return corrs, r2s


# def ridge_for_stacking_lags(train_features, train_targets, test_features, max_L, alphas=[10**j for j in range(-6, 10)], n_splits=5, score_function=r2_multi, return_coefs=False, device="cuda"):
    
#     # predict train, need to manually split (train, test) within train, cv alpha + fit ridge for each split
#     start_time = time.time()
#     kfold = KFold(n_splits=n_splits)
#     train_preds = np.zeros_like(train_targets)
#     for tr_idx, val_idx in kfold.split(train_features):
#         # best alphas
#         train_trn_X = torch.tensor(train_features[tr_idx], dtype=torch.float32, device=device)
#         train_trn_Y = torch.tensor(train_features[tr_idx], dtype=torch.float32, device=device)
#         train_best_alphas = cv_ridge_lags(train_trn_X, train_trn_Y, max_L, alphas=alphas, n_splits=n_splits, print_time=False)
#         # ridge weights
#         train_trn_X = nn.functional.pad(train_trn_X, pad=(0, 0, 0, max_L))
#         train_trn_Y = nn.functional.pad(train_trn_Y, pad=(0, 0, 0, max_L))
#         W = _fit_ridge_per_target_alpha_lags(train_trn_X, train_trn_Y, max_L, train_best_alphas)
#         # predict val
#         train_val_X = torch.tensor(train_features[tr_idx], dtype=torch.float32, device=device)
#         train_val_X = nn.functional.pad(train_val_X, pad=(0, 0, 0, max_L))
#         train_preds[val_idx] = get_pred_lags(train_val_X, W, max_L)[:-max_L].cpu().numpy()
#         # save memory
#         del train_trn_X, train_trn_Y, train_val_X, W
#         torch.cuda.empty_cache()
#     print(f"Time taken for predicting train: {time.time() - start_time:.2f} sec")
    
#     # get the score on the training set
#     train_err = train_targets - train_preds
#     train_scores = score_function(train_targets, train_preds)
#     train_variances = np.var(train_preds, axis=0)
    
#     # then, for all train, cv alpha + fit ridge
#     train_features = torch.tensor(train_features, dtype=torch.float32, device=device)
#     train_targets = torch.tensor(train_targets, dtype=torch.float32, device=device)
#     test_features = torch.tensor(test_features, dtype=torch.float32, device=device)
#     best_alphas = cv_ridge_lags(train_features, train_targets, max_L, alphas=alphas, n_splits=n_splits)
#     train_features = nn.functional.pad(train_features, pad=(0, 0, 0, max_L))
#     train_targets = nn.functional.pad(train_targets, pad=(0, 0, 0, max_L))
#     weights = _fit_ridge_per_target_alpha_lags(train_features, train_targets, max_L, best_alphas)
#     test_features = nn.functional.pad(test_features, pad=(0, 0, 0, max_L))
#     test_preds = get_pred_lags(test_features, weights, max_L)[:-max_L].cpu().numpy()
#     weights_cpu = weights.cpu().numpy()
#     del train_features, train_targets, test_features, weights  # save GPU memory
#     torch.cuda.empty_cache()

#     if return_coefs:
#         return train_preds, train_err, test_preds, train_scores, train_variances, weights_cpu
#     else:
#         return train_preds, train_err, test_preds, train_scores, train_variances


def stacking_lags(train_data, test_data, train_feature, test_feature, max_L, score_f=r2_multi, alphas=[10**j for j in range(-6, 10)], return_ridges=False, device="cuda"):
    """
    Stack on each lag, but don't need to input the lags.
    """
    # [bj] Note: r2_multi argument order is reversed than R2!
    # normalize data by TRAIN/TEST
    train_data = np.nan_to_num(zscore(train_data))
    test_data = np.nan_to_num(zscore(test_data))
    train_feature = torch.from_numpy(np.nan_to_num(zscore(train_feature)))
    test_feature = torch.from_numpy(np.nan_to_num(zscore(test_feature)))
    
    # Number of time points in the test set
    train_data = np.pad(train_data, ((0, max_L), (0, 0)))
    test_data = np.pad(test_data, ((0, max_L), (0, 0)))
    # train_feature = np.pad(train_feature, ((0, max_L), (0, 0)))
    # test_feature = np.pad(test_feature, ((0, max_L), (0, 0)))
    train_feature = nn.functional.pad(train_feature, pad=(0, 0, 0, max_L))
    test_feature = nn.functional.pad(test_feature, pad=(0, 0, 0, max_L))
    n_time_test = test_data.shape[0]

    # Check that the number of voxels is the same in the training and test sets
    assert train_data.shape[1] == test_data.shape[1]
    n_voxels = train_data.shape[1]
    n_features = max_L + 1

    # # Array to store R2 scores for each feature space and voxel
    # r2s = np.zeros((n_features, n_voxels))
    # Array to store R2 scores for each feature space and voxel in the training set
    r2s_train = np.zeros((n_features, n_voxels))
    # Array to store variance explained by the model for each feature space and voxel in the training set
    var_train = np.zeros((n_features, n_voxels))
    # # Array to store R2 scores for each feature space weighted by stacking weights
    # r2s_weighted = np.zeros((n_features, n_voxels))

    # # Array to store stacked predictions for each voxel
    # stacked_pred = np.zeros((n_time_test, n_voxels))
    # # Dictionary to store predictions for each feature space and voxel in the training set
    # preds_train = {}
    # # Dictionary to store predictions for each feature space and voxel in the test set
    # preds_test = np.zeros((n_features, n_time_test, n_voxels))
    # # Array to store weighted predictions for each feature space and voxel in the test set
    # weighted_pred = np.zeros((n_features, n_time_test, n_voxels))
    
    # initialize an error dictionary to store errors for each feature
    err = dict()

    # iterate over each feature and train a model using feature ridge regression
    weights = []
    for FEATURE in range(n_features):
        (
            # preds_train[FEATURE],
            _,
            error,
            # preds_test[FEATURE, :, :],
            _,
            r2s_train[FEATURE, :],
            var_train[FEATURE, :],
            W,
        ) = ridge_for_stacking_torch(
            get_lagged_X(train_feature, FEATURE, max_L).numpy(),
            train_data,
            get_lagged_X(test_feature, FEATURE, max_L).numpy(),
            alphas=alphas,
            n_splits=5,
            return_coefs=True,
            device=device,
        )
        err[FEATURE] = error
        weights.append(W)

    # calculate error matrix for stacking
    P = np.zeros((n_voxels, n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            P[:, i, j] = np.mean(err[i] * err[j], 0)

    # solve the quadratic programming problem to obtain the weights for stacking
    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    S = np.zeros((n_voxels, n_features))
    # stacked_pred_train = np.zeros_like(train_data)

    for i in range(0, n_voxels):
        PP = matrix(P[i])
        # solve for stacking weights for every voxel
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b, show_progress=False)["x"]).reshape(n_features)

        # # combine the predictions from the individual feature spaces for voxel i
        # z_test = np.array([preds_test[feature_j, :, i] for feature_j in range(n_features)])
        # z_train = np.array([preds_train[feature_j][:, i] for feature_j in range(n_features)])
        # # multiply the predictions by S[i,:]
        # stacked_pred[:, i] = np.dot(S[i, :], z_test)
        # # combine the training predictions from the individual feature spaces for voxel i
        # stacked_pred_train[:, i] = np.dot(S[i, :], z_train)

    # # compute the R2 score for the stacked predictions on the training data
    # stacked_train_r2s = score_f(train_data, stacked_pred_train)

    # # compute the R2 scores for each individual feature and the weighted feature predictions
    # for FEATURE in range(n_features):
    #     # weight the predictions according to S:
    #     # weighted single feature space predictions, computed over a fold
    #     weighted_pred[FEATURE, :] = preds_test[FEATURE, :] * S[:, FEATURE]

    # for FEATURE in range(n_features):
    #     r2s[FEATURE, :] = score_f(test_data, preds_test[FEATURE])
    #     r2s_weighted[FEATURE, :] = score_f(test_data, weighted_pred[FEATURE])

    # # compute the R2 score for the stacked predictions on the test data
    # stacked_r2s = score_f(test_data, stacked_pred)

    # # return the results
    # if return_ridges:
    #     return r2s, stacked_r2s, r2s_weighted, r2s_train, stacked_train_r2s, S, weights
    # else:
    #     return r2s, stacked_r2s, r2s_weighted, r2s_train, stacked_train_r2s, S
    
    # return the results
    if return_ridges:
        return r2s_train, S, weights
    else:
        return r2s_train, S