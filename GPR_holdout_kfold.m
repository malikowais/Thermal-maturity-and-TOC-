clc
clear
[num,txt,raw] = xlsread('wl_TOC.xlsx');

input = num(:,5:8);
output = num(:,10);
% if not normlaized then use the following code
factor = max([max(input) abs(min(input))]);
input_norm = input/factor;

[genre, ~, index] = unique(txt(2:end,8));
lithology = logical(accumarray([(1:numel(index)).' index], 1));

factor = max([max(output) abs(min(output))]);
output_norm = output/factor;
all_input = horzcat(input_norm,lithology);
all_norm = horzcat(input_norm,lithology,output_norm);

RegTreeTemp = templateTree('Surrogate','On');

for j=1:100
 
[m,n] = size(all_norm) ;
P = 0.75 ;
idx = randperm(m)  ;
trainingj = all_norm(idx(1:round(P*m)),:) ; 
testingj = all_norm(idx(round(P*m)+1:end),:) ;

%  CV = cvpartition(output_norm,'Holdout',1/4);
%  trIdxj = CV.training;
%  teIdxj = CV.test;
%     
% trainingj = horzcat(all_input(trIdxj,:),output_norm(trIdxj,:));
% testingj = horzcat(all_input(teIdxj,:),output_norm(teIdxj,:)); 



c = cvpartition(trainingj(:,8),'KFold',10);

index=1;
T = [trainingj(:,1) trainingj(:,2) trainingj(:,3) trainingj(:,4) trainingj(:,5) trainingj(:,6) trainingj(:,7)];



for i = 1:c.NumTestSets
    trIdx = c.training(i);
    teIdx = c.test(i);
    
    training = horzcat(trainingj(trIdx,1:7),trainingj(trIdx,8));
    testing = horzcat(trainingj(teIdx,1:7),trainingj(teIdx,8));
    
 mdl = fitrgp(training(:,1:7),training(:,8));
     
     y_training = training(:,8);
    Predicted_training = resubPredict(mdl);
    [r2 rmse1] = rsquare(y_training,Predicted_training);
    r_training(i) = r2;
    L_training(i) = rmse1;

    y = testing(:,8);
    Predicted = predict(mdl,testing(:,1:7));
    [r2 rmse1] = rsquare(y,Predicted);
    r_testing(i) = r2;
    L_testing(i) = rmse1;
end

X(j,:) = horzcat(mean(r_training),mean(r_testing),mean(L_training),mean(L_testing));

mdl_j = fitrgp(trainingj(:,1:7),trainingj(:,8));
 y_training = trainingj(:,8);
    Predicted_training = resubPredict(mdl_j);
    [r2 rmse1] = rsquare(y_training,Predicted_training);
    r_training_j(j) = r2;
    L_training_j(j) = rmse1;
    MAE_training(j) = errperf(y_training,Predicted_training,'mae'); 
    MAPE_training(j) = errperf(y_training,Predicted_training,'mape');
    MARE_training(j) = errperf(y_training,Predicted_training,'mare');
    y = testingj(:,8);
    Predicted = predict(mdl_j,testingj(:,1:7));
    [r2 rmse1] = rsquare(y,Predicted);
    r_testing_j(j) = r2;
    L_testing_j(j) = rmse1;
    MAE_testing(j) = errperf(y,Predicted,'mae');
    MAPE_testing(j) = errperf(y,Predicted,'mape');
    MARE_testing(j) = errperf(y,Predicted,'mare');
    
    
end
Y = horzcat(mean(r_training_j),mean(r_testing_j),mean(L_training_j),mean(L_testing_j));
Y_std = horzcat(std(r_training_j),std(r_testing_j),std(L_training_j),std(L_testing_j));

M = horzcat(mean(MAE_training),mean(MAPE_training),mean(MARE_training),mean(MAE_testing),mean(MAPE_testing),mean(MARE_testing));


