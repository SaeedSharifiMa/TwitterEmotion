pval = zeros(1,size(X_train_bag,2));
for i=1:size(X_train_bag,2)
    [table,chi2,pval(i)] = crosstab(Y_train, X_train_bag(:,i));
end
cutoff = 0.2; % p-value cutoff
extractedFeatures = find(pval <= cutoff); % = 3039 features are selected
%validate(Naive_Bayes_Classifier, X_train_bag(:,extractedFeatures), train_raw, Y_train, 5) % = 0.9102
%validate(Random_Forest_Classifier, X_train_bag(:,extractedFeatures), train_raw, Y_train, 5) % = 1.0276 with 100 trees