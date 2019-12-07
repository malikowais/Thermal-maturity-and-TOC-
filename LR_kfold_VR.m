clc
clear
[num,txt,raw] = xlsread('gc_wl_VR_noCoal.xlsx');

input = num(:,5:9);
output = num(:,13);
% if not normlaized then use the following code
factor = max([max(input) abs(min(input))]);
input_norm = input/factor;

%Not using lithology as well as S1 and S2
% [genre, ~, index] = unique(txt(2:end,12));
% lithology = logical(accumarray([(1:numel(index)).' index], 1));

factor = max([max(output) abs(min(output))]);
output_norm = output/factor;
% all_input = horzcat(input_norm,lithology);
all_input = input_norm;
% all_norm = horzcat(input_norm,lithology,output_norm);
all_norm = horzcat(input_norm,output_norm);


c = cvpartition(output_norm,'KFold',10);

index=1;

Y=0;
Z=0;

for i = 1:c.NumTestSets
    trIdx = c.training(i);
    teIdx = c.test(i);
    
    training = horzcat(all_input(trIdx,:),output_norm(trIdx,:));
    testing = horzcat(all_input(teIdx,:),output_norm(teIdx,:));
    
     mdl = fitlm(all_input(trIdx,:),output_norm(trIdx,:));
    
    y_training = output_norm(trIdx,:);
    Predicted_training = predict(mdl);
    [r2 rmse1] = rsquare(y_training,Predicted_training);
    r_training(i) = r2;
    L_training(i) = rmse1;
    [r2 RMSE_Training(i)] = rsquare(y_training*factor,Predicted_training*factor); % VR actual value
    MAE_training(i) = errperf(y_training,Predicted_training,'mae'); 
    MAPE_training(i) = errperf(y_training,Predicted_training,'mape');
    MARE_training(i) = errperf(y_training,Predicted_training,'mare');
    MAE_training_VR(i) = errperf(y_training*factor,Predicted_training*factor,'mae'); 
    y = output_norm(teIdx,:);
    Predicted = predict(mdl,all_input(teIdx,:));
    [r2 rmse1] = rsquare(y,Predicted);
    r_testing(i) = r2;
    L_testing(i) = rmse1;
    [r2 RMSE_Testing(i)] = rsquare(y*factor,Predicted*factor); % VR actual value
    MAE_testing(i) = errperf(y,Predicted,'mae');
    MAPE_testing(i) = errperf(y,Predicted,'mape');
    MARE_testing(i) = errperf(y,Predicted,'mare');
    MAE_testing_VR(i) = errperf(y*factor,Predicted*factor,'mae');
    
    Y=vertcat(Y,y);
    Z=vertcat(Z,Predicted);
end

X = horzcat(mean(r_training),mean(r_testing),mean(L_training),mean(L_testing));

M = horzcat(mean(MAE_training),mean(MAPE_training),mean(MARE_training),mean(MAE_testing),mean(MAPE_testing),mean(MARE_testing));

MAE_tr_All = mean(MAE_training_VR);
MAE_ts_All = mean(MAE_testing_VR);
RMSE_tr_All = mean(RMSE_Training);
RMSE_ts_All = mean(RMSE_Testing);
OUTPUT = horzcat(RMSE_tr_All, mean(r_training),MAE_tr_All, RMSE_ts_All, mean(r_testing),MAE_ts_All)