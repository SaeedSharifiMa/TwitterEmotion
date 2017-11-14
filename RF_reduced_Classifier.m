classdef RF_reduced_Classifier < handle
   properties
      rf_model,
      X_train_mean,
      not_sparse_features,
      X_train_mean_not_sparse,
      coeff
   end
   methods
      function train(obj, X_train_bag, train_raw, Y_train)
        load('validation.mat');
        costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
        
        [n_train, p] = size(X_train_bag);
        obj.not_sparse_features = sum(X_train_bag) > 3;
        X_combined = [X_train_bag; X_validation_bag];
        
        obj.X_train_mean = mean(X_combined);

        
        obj.X_train_mean_not_sparse = obj.X_train_mean(obj.not_sparse_features);
        X_train_not_sparse = X_train_bag(:, obj.not_sparse_features);
        
        obj.X_combined_not_sparse = X_combined(:, obj.not_sparse_features);
        
       
        
        [coeff, score, latenet] = pca(full(X_combined));
        obj.coeff = coeff;
        
        X_train_not_sparse_centered = X_train_not_sparse - repmat(obj.X_train_mean_not_sparse, n_train, 1);
        X_train_not_sparse_pca = X_train_not_sparse_centered * coeff(:,:);
        
        obj.rf_model = TreeBagger(50, X_train_not_sparse_pca, Y_train, 'Cost', costs);
      end
      
      function y_pred = predict(obj, X_test_bag, test_raw)
          n_test = size(X_test_bag, 1);
          X_test_not_sparse = X_test_bag(:, obj.not_sparse_features);
          X_test_not_sparse_centered = X_test_not_sparse - repmat(obj.X_train_mean_not_sparse, n_test, 1);
          X_test_not_sparse_pca = X_test_not_sparse_centered * obj.coeff(:,:);
          
          y_pred_str = predict(obj.rf_model, X_test_not_sparse_pca);
          y_pred = str2num(cell2mat(y_pred_str));
      end
   end
end