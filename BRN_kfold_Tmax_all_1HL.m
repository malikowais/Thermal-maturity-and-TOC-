
clc
clear
[num,txt,raw] = xlsread('gc_wl_Tmax_All.xlsx');

input = num(:,4:8);
output = num(:,12);
% if not normlaized then use the following code
factor = max([max(input) abs(min(input))]);
input_norm = input/factor;

%Not using lithology as well as S1 and S2
% [genre, ~, index] = unique(txt(2:end,11));
% lithology = logical(accumarray([(1:numel(index)).' index], 1));

factor = max([max(output) abs(min(output))]);
output_norm = output/factor;
% all_input = horzcat(input_norm,lithology);
all_input = input_norm;
% all_norm = horzcat(input_norm,lithology,output_norm);
all_norm = horzcat(input_norm,output_norm);
% Hidden_Layer_Best= zeros(400,6);
indexk=1;
prev_tr_best = 0;
prev_ts_best = 0;
for k=2:20
    
for j = 1:50
c = cvpartition(output_norm,'KFold',10);



Y=0;
Z=0;


% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.

% Create a Fitting Network
hiddenLayerSize = k;
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = '';  % Divide data randomly
% net.divideMode = 'sample';  % Divide up every sample
% net.divideParam.trainRatio = 80/100;
% net.divideParam.valRatio = 10/100;
% net.divideParam.testRatio = 10/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

for i = 1:c.NumTestSets
    trIdx = c.training(i);
    teIdx = c.test(i);
    
    training = horzcat(all_input(trIdx,:),output_norm(trIdx,:));
    testing = horzcat(all_input(teIdx,:),output_norm(teIdx,:));
    
    % Train the Network
    [net,tr] = train(net,all_input(trIdx,:)',output_norm(trIdx,:)');
 

    y_training = output_norm(trIdx,:);
    % Test the Network
    Predicted_training = net(all_input(trIdx,:)');
    [r2 rmse1] = rsquare(y_training,Predicted_training');
    r_training(i) = r2;
    L_training(i) = rmse1;
    [r2 RMSE_Training(i)] = rsquare(y_training*factor,Predicted_training'*factor); % TOC actual value
    MAE_training(i) = errperf(y_training,Predicted_training','mae'); 
    MAPE_training(i) = errperf(y_training,Predicted_training','mape');
    MARE_training(i) = errperf(y_training,Predicted_training','mare');
    MAE_training_TOC(i) = errperf(y_training*factor,Predicted_training'*factor,'mae'); 
    y = output_norm(teIdx,:);
    Predicted = net(all_input(teIdx,:)');
    [r2 rmse1] = rsquare(y,Predicted');
    r_testing(i) = r2;
    L_testing(i) = rmse1;
    [r2 RMSE_Testing(i)] = rsquare(y*factor,Predicted'*factor); % TOC actual value
    MAE_testing(i) = errperf(y,Predicted','mae');
    MAPE_testing(i) = errperf(y,Predicted','mape');
    MARE_testing(i) = errperf(y,Predicted','mare');
    MAE_testing_TOC(i) = errperf(y*factor,Predicted'*factor,'mae');
    
    Y=vertcat(Y,y);
    Z=vertcat(Z,Predicted');
end

X = horzcat(mean(r_training),mean(r_testing),mean(L_training),mean(L_testing));

M = horzcat(mean(MAE_training),mean(MAPE_training),mean(MARE_training),mean(MAE_testing),mean(MAPE_testing),mean(MARE_testing));


MAE_tr_All = mean(MAE_training_TOC);
MAE_ts_All = mean(MAE_testing_TOC);
RMSE_tr_All = mean(RMSE_Training);
RMSE_ts_All = mean(RMSE_Testing);

OUTPUT = horzcat(RMSE_tr_All, mean(r_training),MAE_tr_All, RMSE_ts_All, mean(r_testing),MAE_ts_All);
All_OUTPUT(j,:) = OUTPUT;

YY(j,:)=Y*factor;
ZZ(j,:)=Z*factor;

if (mean(r_testing) > prev_ts_best) && (mean(r_training) > prev_tr_best)
    gnet = net;
    prev_ts_best = mean(r_testing);
    prev_tr_best = mean(r_training);
    BestYY = YY(j,:);
    BestZZ = ZZ(j,:);
end
end
Hidden_Layer(indexk,:) = mean(All_OUTPUT);
[mm,ii] =max(All_OUTPUT(:,5));
best = All_OUTPUT(ii,:);
Hidden_Layer_Best(indexk,:) = best(1,:);
indexk = indexk + 1;
end

