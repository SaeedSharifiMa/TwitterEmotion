function validation_error = validate(train, X, X_raw, Y, num_folds)
    partitions = cvpartition(num_points,'KFold',num_folds);
    costs = zeros(num_folds, 1);
    for fold_i = 1:num_folds
        train_indxs = partitions.training(fold_i);
        test_indxs = partition.test(fold_i);
        
        X_train = X(train_indxs, :);
        X_train_raw = X_raw(train_indxs, :);
        Y_train = Y(train_indxs, :);
        
        X_test = X(test_indxs, :);
        X_test_raw = X_raw(train_indxs, :);
        Y_test = Y(test_indxs, :);
        
        model = train(X_train, X_train_raw, Y_train);
        
        Y_test_predicted = model.predict(X_test, X_test_raw);
        
        curr_cost = performance_measure(Y_test_predicted, Y_test);
        costs(fold_i) = curr_cost;
    end
    
   validation_error = mean(cost);
end