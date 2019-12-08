
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


 c = cvpartition(output,'KFold',10);
index=1;
T_double = [all_input(:,1) all_input(:,2) all_input(:,3) all_input(:,4) all_input(:,5) all_input(:,6) all_input(:,7) output_norm ];
T = array2table(T_double);

kk=1; %index to store all testing results
Y=0;
Z=0;
for j = 1:300
    maxMinLS = 20;
    minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
    numPTS = optimizableVariable('numPTS',[1,size(T,2)-1],'Type','integer');
    hyperparametersRF = [minLS; numPTS];
    results = bayesopt(@(params)oobErrRF(params,T,j,output_norm),hyperparametersRF, 'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);
    bestOOBErr = results.MinObjective;
    bestHyperparameters = results.XAtMinObjective;
%     close all    
for i = 1:c.NumTestSets
    trIdx = c.training(i);
    teIdx = c.test(i);
    
    training = horzcat(all_input(trIdx,:),output_norm(trIdx,:));
    testing = horzcat(all_input(teIdx,:),output_norm(teIdx,:));
   B = TreeBagger(j,all_input(trIdx,:),output_norm(trIdx,:),'Method','regression','MinLeafSize',bestHyperparameters.minLS,'NumPredictorstoSample',bestHyperparameters.numPTS,'OOBPredictorImportance','on');


    y_training = output_norm(trIdx,:);
    Predicted_training=predict(B,all_input(trIdx,:));

     [r2 rmse1] = rsquare(y_training,Predicted_training);
    r_training(i) = r2;
     L_training(i) = rmse1;
     [r2 RMSE_Training(i)] = rsquare(y_training*factor,Predicted_training*factor); % TOC actual value
   
    MAE_training(i) = errperf(y_training,Predicted_training,'mae'); 
    MAPE_training(i) = errperf(y_training,Predicted_training,'mape');
    MARE_training(i) = errperf(y_training,Predicted_training,'mare');
    MAE_training_TOC(i) = errperf(y_training*factor,Predicted_training*factor,'mae'); 
    y = output_norm(teIdx,:);
    Predicted=predict(B,all_input(teIdx,:));

    [r2 rmse1] = rsquare(y,Predicted);
    r_testing(i) = r2;
    L_testing(i) = rmse1;
    [r2 RMSE_Testing(i)] = rsquare(y*factor,Predicted*factor); % TOC actual value
    MAE_testing(i) = errperf(y,Predicted,'mae');
    MAPE_testing(i) = errperf(y,Predicted,'mape');
    MARE_testing(i) = errperf(y,Predicted,'mare');
    MAE_testing_TOC(i) = errperf(y*factor,Predicted*factor,'mae'); 
    Y=vertcat(Y,y);
    Z=vertcat(Z,Predicted);

end

X(index,:) = horzcat(mean(r_training),mean(r_testing),mean(L_training),mean(L_testing));
M(index,:) = horzcat(mean(MAE_training),mean(MAPE_training),mean(MARE_training),mean(MAE_testing),mean(MAPE_testing),mean(MARE_testing));
M_ALL(index,:) = horzcat(mean(MAE_training_TOC),mean(MAE_testing_TOC),mean(RMSE_Training),mean(RMSE_Testing));
index = index + 1;
end

imp = B.OOBPermutedPredictorDeltaError;
c = categorical({'GR',	'RHOB',	'DTC',	'RT',	'Lith-Coal', 'Lith-Mudstone', 'Lith-ShalyCoal'});
figure;
bar(c,imp);
% title('Curvature Test');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
% h.XTickLabel =B.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% view(B.Trees{1},'Mode','graph')

Y= Y*factor;
Z= Z*factor;
YZ=horzcat(Y,Z);
YZ = sortrows(YZ,1);
XX= 1:67;
scatter(XX,YZ(2:end,1));
hold on
scatter(XX,YZ(2:end,2));
MAE_tr_All = mean(MAE_training_TOC)
MAE_ts_All = mean(MAE_testing_TOC)
RMSE_tr_All = mean(RMSE_Training)
RMSE_ts_All = mean(RMSE_Testing)

