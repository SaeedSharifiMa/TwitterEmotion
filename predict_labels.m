function [Y_hat] = predict_labels(X_test_bag, test_raw)
% Inputs:   X_test_bag     nx10000 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)
    load('train.mat');
    nb_classifier = Naive_Bayes_Classifier;
    nb_classifier.train(X_train_bag, train_raw, Y_train);
    Y_hat = nb_classifier.predict(X_test_bag, test_raw);
end
