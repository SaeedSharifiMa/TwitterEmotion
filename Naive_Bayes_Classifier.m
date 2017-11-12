classdef Naive_Bayes_Classifier < handle
   properties
      nb_model
   end
   methods
      function train(obj, X_train_bag, train_raw, Y_train)
         obj.nb_model = fitcnb(X_train_bag, Y_train, 'Distribution','mn');
      end
      
      function y_pred = predict(obj, X_test_bag, test_raw)
          
          y_pred = predict(obj.nb_model, X_test_bag);
      end
   end
end