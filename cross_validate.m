function cross_validation_error = cross_validate(train_model, labels, word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets, num_folds)
% Inputs:   word_counts     nx10000 word counts features
%           cnn_feat        nx4096 Penultimate layer of Convolutional
%                               Neural Network features
%           prob_feat       nx1365 Probabilities on 1000 objects and 365
%                               scene categories
%           color_feat      nx33 Color spectra of the images (33 dim)
%           raw_imgs        nx30000 raw images pixels
%           raw_tweets      nx1 cells containing all the raw tweets in text
    num_points = size(word_counts, 1);
    partitions = cvpartition(num_points,'KFold',num_folds);
    
    costs = zeros(num_folds, 1);
    
    for fold_i = 1:num_folds
        train_indxs = partitions.training(fold_i);
        test_indxs = partition.test(fold_i);
        
        %word_counts
        train_word_counts = word_counts(train_indxs, :);
        test_word_counts = word_counts(test_indxs, :);
        
        %cnn_feat
        train_cnn_feat = cnn_feat(train_indxs, :);
        test_cnn_feat = cnn_feat(test_indxs, :);
        
        %prob_feat
        train_prob_feat = prob_feat(train_indxs, :);
        test_prob_feat = prob_feat(test_indxs, :);
        
        %color_feat
        train_color_feat = color_feat(train_indxs, :);
        test_color_feat = color_feat(test_indxs, :);
        
        %raw_imgs
        train_raw_imgs = raw_imgs(train_indxs, :);
        test_raw_imgs = raw_imgs(test_indxs, :);
        
        %raw_tweets
        train_raw_tweets = raw_tweets(train_indxs, :);
        test_raw_tweets = raw_tweets(test_indxs, :);
        
        %train_model model
        model = train_model(train_word_counts, train_cnn_feat, train_prob_feat, train_prob_feat, train_color_feat, train_raw_imgs, train_raw_tweets);
        predicted_labels = model.predict(test_word_counts, test_cnn_feat, test_prob_feat, test_color_feat, test_raw_imgs, test_raw_tweets);
        true_labels = labels(test_indxs);
        
        %calculate the total cost
        costs(fold_i) = sum(predicted_labels ~= true_labels); %TODO: FIX THIS
    end
    
    cross_validation_error = mean(costs);
end