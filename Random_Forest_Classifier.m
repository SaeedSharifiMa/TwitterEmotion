classdef Random_Forest_Classifier < handle
   properties
      rf_model
   end
   methods
      function train(obj, X_train_bag, train_raw, Y_train)
        costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

         obj.rf_model = TreeBagger(50, full(X_train_bag), Y_train, 'Cost', costs);
      end
      
      function y_pred = predict(obj, X_test_bag, test_raw)
          y_pred_str = predict(obj.rf_model, full(X_test_bag));
          y_pred = str2num(cell2mat(y_pred_str));
      end
   end
end