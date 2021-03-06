classdef Naive_Bayes_Classifier < handle
   properties
      nb_model
   end
   methods
      function train(obj, X_train_bag, train_raw, Y_train)
        costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

         obj.nb_model = fitcnb(X_train_bag, Y_train, 'Distribution','mn','Cost', costs);
      end
      
      function y_pred = predict(obj, X_test_bag, test_raw)
          y_pred = predict(obj.nb_model, X_test_bag);
      end
   end
end