function [Y_hat] = predict_labels(X_test_bag, test_raw)
% Inputs:   X_test_bag     nx10000 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)
    load('train.mat');
    [n_train,p] = size(X_train_bag);
    n_test = size(X_test_bag, 1);
    X_train_mean = mean(X_train_bag);
    
    not_sparse_features = X_train_mean > 10./n_train;
    
    X_train_mean_not_sparse = X_train_mean(not_sparse_features);
    X_train_not_sparse = X_train_bag(:, not_sparse_features);
    X_test_not_sparse = X_test_bag(:, not_sparse_features);
    
    [coeff, score, latenet] = pca(full(X_train_not_sparse));
    
    X_train_not_sparse_centered = X_train_not_sparse - repmat(X_train_mean_not_sparse, n_train, 1);
    X_test_not_sparse_centered = X_test_not_sparse - repmat(X_train_mean_not_sparse, n_test, 1);
    
    X_train_not_sparse_pca = X_train_not_sparse_centered * coeff;
    X_test_not_sparse_pca = X_test_not_sparse_centered * coeff;
    rf_classifier = Random_Forest_Classifier;
    
    rf_classifier.train(X_train_not_sparse_pca, train_raw, Y_train);
    Y_hat = rf_classifier.predict(X_test_not_sparse_pca, test_raw);
end
