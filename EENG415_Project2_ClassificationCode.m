%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clayton Daly, Jesse Dugan, Holly Hammons, Luke Logan
% EENG 415
% Dr. Salman Mohagheghi 
% 5/6/2022
% Project 2: Classification Code
% NOTE: you will need the file "contains_clustering_outcomes.csv"
% in the same directory to run this file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all
%% KNN Classifier 
data = readtable('contains_clustering_outcomes.csv');
% data.class = randi(3,365,1);

% Separate training and testing data
% [train,test] = crossvalind('LeaveMOut',height(data),31);
train = logical([ones(1,334) zeros(1,31)]');
test = logical([zeros(1,334) ones(1,31)]');

training = data(train,2:end);
testing = data(test,2:end);

% Find the optimal value of k by testing different values from 1 to N
N = 20;
perfRecord = zeros(N,4);
for k=1:N
    knn_mdl = fitcknn(training,'Class','NumNeighbors',k);
    class_out = predict(knn_mdl,testing(:,1:end-1));
    
    % Calculate performance metrics for classifier
    cp = classperf(testing.Class,class_out);
    [c_matrix,order] = confusionmat(testing.Class,class_out);
    [r,c] = size(c_matrix);
    sensitivity = ones(r,1);
    specificity = ones(r,1);
    for i = 1:r
        if sum(c_matrix(i,:)) > 0
            sensitivity(i) = c_matrix(i,i) / sum(c_matrix(i,:));
        else
            sensitivity(i) = 1;
        end
        specificity(i) = (sum(c_matrix,'all') - sum(c_matrix(i,:)) - sum(c_matrix(:,i)) + c_matrix(i,i)) / (sum(c_matrix,'all') - sum(c_matrix(i,:)));
    end
    % Define the weights of classification parameters 
    wsens = 0.3;
    wspec = 0.3;
    wcorr = 0.4;
    
    % Find score based on weights of parameters
    sens = sum(sensitivity)/length(sensitivity);
    spec = sum(specificity)/length(specificity);
    perfRecord(k,1) = cp.CorrectRate;
    perfRecord(k,2) = sens;
    perfRecord(k,3) = spec;
    perfRecord(k,4) = wcorr*cp.CorrectRate + wsens*sens + wspec*spec;
end

% Output the best value of k based on a score
[score,best_k] = max(perfRecord(:,4));
knn_mdl = fitcknn(training,'Class','NumNeighbors',best_k);
[class_out,KNNscores] = predict(knn_mdl,testing(:,1:end-1));
evaluate_classifier(testing.Class,class_out,"KNN")
roccurve(KNNscores,testing.Class,'KNN')
%% Logistic Regression
% Fit a logistic model to the dataset
X = table2array(training(:,1:end-1));
Y = categorical(training.Class);
B = mnrfit(X,Y);

% Find the classes predicted for each customer
test_set = table2array(testing(:,1:end-1));
pihat = mnrval(B,test_set);
cat = categories(Y);
class_out = ones(length(pihat),1);
for i=1:length(pihat)
    [~,idx] = max(pihat(i,:));
    class_out(i) = string(cat(idx));
end

evaluate_classifier(testing.Class,class_out,"Logistic")
roccurve(pihat,testing.Class,'Logistic Regression')
%% Linear Discriminant Analyisis
lda_mdl = fitcdiscr(training,'Class','DiscrimType','linear');
[class_out,LDAscores] = predict(lda_mdl,testing(:,1:end-1));
evaluate_classifier(testing.Class,class_out,"LDA")
roccurve(LDAscores,testing.Class,'LDA')
%% Quadratic Discriminant Analyisis
lda_mdl = fitcdiscr(training,'Class','DiscrimType','quadratic');
[class_out,QDAscores] = predict(lda_mdl,testing(:,1:end-1));
evaluate_classifier(testing.Class,class_out,"QDA")
roccurve(QDAscores,testing.Class,'QDA')
%% Classification Tree
tree_mdl = fitctree(training,'Class');
[class_out, CTscores] = predict(tree_mdl,testing(:,1:end-1));
evaluate_classifier(testing.Class,class_out,"Tree")
roccurve(CTscores,testing.Class,'Classification Tree')
%% Function to Evaluate Classifier Performance
function evaluate_classifier(test,class_out,name)

    % Calculate performance metrics for classifier
    cp = classperf(test,class_out);
    % Find the confusion matrix
    [c_matrix,order] = confusionmat(test,class_out);
    % If the confusion matrix is missing class 1, add it
    if (length(c_matrix) == 3)
        c_matrix = [zeros(3,1) c_matrix];
        c_matrix = [zeros(1,4)' c_matrix']';
    end
    % Display the confusion matrix
    figure
    confusionchart(c_matrix)
    title([name "% Classifier"])
    
    % Calculate sensitivity and specificity from the confusion matrix
    [r,c] = size(c_matrix);
    sensitivity = ones(r,1);
    specificity = ones(r,1);
    for i = 1:r
        if sum(c_matrix(i,:)) > 0
            % Sensitivity = TP / P
            sensitivity(i) = c_matrix(i,i) / sum(c_matrix(i,:));
        else
            % If there are no instances of a class
            sensitivity(i) = 1;
        end
        % Specifitity = TN / N
        specificity(i) = (sum(c_matrix,'all') - sum(c_matrix(i,:)) - sum(c_matrix(:,i)) + c_matrix(i,i)) / (sum(c_matrix,'all') - sum(c_matrix(i,:)));
    end
    % Find the total sensitivity and specificity based on the sum of
    % sensitivity and specificity for each class
    sens = sum(sensitivity)/length(sensitivity);
    spec = sum(specificity)/length(specificity);
    
    % Define the weights of classification parameters 
    wsens = 0.3;
    wspec = 0.3;
    wcorr = 0.4;
    
    % Calculate the score based on weighted metrics
    score = wcorr*cp.CorrectRate + wsens*sens + wspec*spec;
    fprintf("%s Classifier %s:\n",name);
    fprintf("\t\tScore = %f\n",score);
end

%% Function to plot ROC curve
function roccurve(probability_scores,true_class,modeltype)
% Inputs
% probability_scores: N x C matrix containing predicted probabilities that
% each sample belongs to a certain class, where N is number of samples and
% C is number of classes
% true_class: actual class that each test sample belongs to found in
% clustering algorithm
% modeltype: string input of type of model used to get probability scores

diffscore1 = zeros(1,31);
diffscore2 = zeros(1,31);
diffscore3 = zeros(1,31);

% diffscore is used to find the score of the positive class minus the max
% of the negative scores
for i = 1:31
    temp = probability_scores(i,:);
    
    diffscore1(i) = temp(2) - max([temp(1),temp(3),temp(4)]);
    diffscore2(i) = temp(3) - max([temp(1),temp(2),temp(4)]);
    diffscore3(i) = temp(4) - max([temp(1),temp(2),temp(3)]);
end

% Find True Positive Rate and False Positive Rate for each class, last
% input argument is the class that is being considered positive, and the
% rest are negative
% Class 1 is not used since it doesn't appear in test classes
[X1,Y1,~,AUC1] = perfcurve(true_class,diffscore1,2);   % class 2 +
[X2,Y2,~,AUC2] = perfcurve(true_class,diffscore2,3);   % class 3 +
[X3,Y3,~,AUC3] = perfcurve(true_class,diffscore3,4);   % class 4 +

figure
plot(X1,Y1,'LineWidth',2)
hold on
plot(X2,Y2,'LineWidth',2,'LineStyle','-.')
plot(X3,Y3,'LineWidth',2,'LineStyle',':')
legend(['Class 2 Positive (AUC = ',num2str(AUC1),')'],...
    ['Class 3 Positive (AUC = ',num2str(AUC2),')'],...
    ['Class 4 Positive (AUC = ',num2str(AUC3),')'],'Location','southeast')
title(['ROC Curves for Classes 2,3, and 4 (',modeltype,' Model)'])
xlabel('False Positive Rate')
ylabel('True Positive Rate')
ylim([0,1.05])

end