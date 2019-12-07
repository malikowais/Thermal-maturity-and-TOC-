
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

 c = cvpartition(output,'KFold',10);
index=1;

T_double = [all_input(:,1) all_input(:,2) all_input(:,3) all_input(:,4) all_input(:,5) output_norm ];
T = array2table(T_double);
Y=0;
Z=0;
for j = 10:10:300
%     maxMinLS = 20;
%     minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
%     numPTS = optimizableVariable('numPTS',[1,size(T,2)-1],'Type','integer');
%     hyperparametersRF = [minLS; numPTS];
%     results = bayesopt(@(params)oobErrRF(params,T,j,output_norm),hyperparametersRF, 'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);
%     bestOOBErr = results.MinObjective;
%     bestHyperparameters = results.XAtMinObjective;
%     close all    
for i = 1:c.NumTestSets
    trIdx = c.training(i);
    teIdx = c.test(i);
    
    training = horzcat(all_input(trIdx,:),output_norm(trIdx,:));
    testing = horzcat(all_input(teIdx,:),output_norm(teIdx,:));
    B = TreeBagger(j,all_input(trIdx,:),output_norm(trIdx,:),'Method','regression', 'OOBPredictorImportance','on');
%     B = TreeBagger(j,all_input(trIdx,:),output_norm(trIdx,:),'Method','regression','MinLeafSize',bestHyperparameters.minLS,'NumPredictorstoSample',bestHyperparameters.numPTS);

%     RegTreeEns = fitensemble(input(trIdx,:),output(trIdx,:),'Bag',100,RegTreeTemp, 'Type', 'Regression' );
%     RegTreeEns = fitensemble(input(trIdx,:),output(trIdx,:),'LSBoost',100,RegTreeTemp);
%     Predicted = predict(RegTreeEns,input(teIdx,1));
%     tree = fitrtree(input(trIdx,:),output(trIdx,:));
%     Predicted = predict(tree,input(teIdx,1));
     
%      mdl = fitrsvm(input(trIdx,:),output(trIdx,:),'Standardize',true)
%     mdl = fitrsvm(input(trIdx,:),output(trIdx,:),'KernelFunction','gaussian','Standardize',true);
    
    y_training = output_norm(trIdx,:);
    Predicted_training=predict(B,all_input(trIdx,:));
%     Predicted_training = resubPredict(mdl);
%     r_training(i) = min(min(corrcoef(y_training,Predicted_training)));
%     L_training(i) = resubLoss(mdl);
     [r2 rmse1] = rsquare(y_training,Predicted_training);
    r_training(i) = r2;
     L_training(i) = rmse1;
[r2 RMSE_Training(i)] = rsquare(y_training*factor,Predicted_training*factor); % VR actual value
   
    MAE_training(i) = errperf(y_training,Predicted_training,'mae'); 
    MAPE_training(i) = errperf(y_training,Predicted_training,'mape');
    MARE_training(i) = errperf(y_training,Predicted_training,'mare');
    MAE_training_VR(i) = errperf(y_training*factor,Predicted_training*factor,'mae'); 
%     MSE(i) = mean((y_training - Predicted_training).^2);
    y = output_norm(teIdx,:);
%     Predicted = predict(mdl,input(teIdx,:));
    Predicted=predict(B,all_input(teIdx,:));
%     r_testing(i) = min(min(corrcoef(y,Predicted)));
%     L_testing(i) = loss(mdl,input(teIdx,:),y);
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
%     AllAPE(i) = meanabs(APE);
end
%cvErr = sum(err)/sum(CVO.TestSize);

X(index,:) = horzcat(mean(r_training),mean(r_testing),mean(L_training),mean(L_testing));
M(index,:) = horzcat(mean(MAE_training),mean(MAPE_training),mean(MARE_training),mean(MAE_testing),mean(MAPE_testing),mean(MARE_testing));
M_ALL(index,:) = horzcat(mean(MAE_training_VR),mean(MAE_testing_VR),mean(RMSE_Training),mean(RMSE_Testing));
RESULT(index,:) = horzcat(mean(RMSE_Training),mean(r_training), mean(MAE_training_VR), mean(RMSE_Testing),mean(r_testing), mean(MAE_testing_VR));
index = index + 1;
end

% imp = B.OOBPermutedPredictorDeltaError;
% c = categorical({'GR',	'RHOB',	'DTC',	'RT',	'Lith-Coal', 'Lith-Mudstone', 'Lith-ShalyCoal'});
% figure;
% bar(c,imp);
% % title('Curvature Test');
% ylabel('Predictor importance estimates');
% xlabel('Predictors');
% h = gca;
% % h.XTickLabel =B.PredictorNames;
% h.XTickLabelRotation = 45;
% h.TickLabelInterpreter = 'none';

% view(B.Trees{1},'Mode','graph')



MAE_tr_All = mean(MAE_training_VR)
MAE_ts_All = mean(MAE_testing_VR)
RMSE_tr_All = mean(RMSE_Training)
RMSE_ts_All = mean(RMSE_Testing)
OUTPUT = horzcat(RMSE_tr_All, mean(r_training),MAE_tr_All, RMSE_ts_All, mean(r_testing),MAE_ts_All)
