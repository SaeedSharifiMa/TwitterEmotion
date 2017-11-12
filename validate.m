function validation_error = validate(classifier, X, X_raw, Y, num_folds)
    num_points = size(X, 1);
    partitions = cvpartition(num_points,'KFold',num_folds);
    costs = zeros(num_folds, 1);
    
    for fold_i = 1:num_folds
        train_indxs = partitions.training(fold_i);
        test_indxs = partitions.test(fold_i);
        
        X_train = X(train_indxs, :);
        X_train_raw = X_raw(train_indxs, :);
        Y_train = Y(train_indxs, :);
        
        X_test = X(test_indxs, :);
        X_test_raw = X_raw(train_indxs, :);
        Y_test = Y(test_indxs, :);
        
        
        curr_classifier = classifier;
        curr_classifier.train(X_train, X_train_raw, Y_train);
        
        Y_test_predicted = curr_classifier.predict(X_test, X_test_raw);
        
        curr_cost = performance_measure(Y_test_predicted, Y_test);
        costs(fold_i) = curr_cost;
    end
    
   validation_error = mean(costs);
end