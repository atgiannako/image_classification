%% RF binary classifier
% Giannakopoulos Athanasios
% Kyritsis Georgios

clearvars; close all; clear;

% Load features and labels of training data
load train.mat;

%add path for normalize function
addpath(genpath('/home/kyritsis/myfiles/Machine/code/DeepLearnToolbox-master'));

%Extracting the CNN features and the corresponding labels
%Change to train.X_hog if you want to use HOG features
x = train.X_cnn;
y = train.y;

%Find the indices of airplanes, cars, and horses
%and set the corresponding labels to 1. Set the labels 
%for others to 0
airplane = find(y == 1); 
car = find(y == 2);
horse = find(y == 3);
other = find(y == 4);

y(car) = 1;
y(horse) = 1;
y(other) = 0;

%Converting the labels to categorical
y = categorical(y);

%fix seed
rng(8339);

%Create a random partition for 5-fold cross-validation
%y is a categorical variable 
%Each subsample has roughly the same size 
%and roughly the same class proportions as in y
CVO = cvpartition(y,'KFold',5);

%Initializing vectors for holding the Test and Train errors for each k-fold
errorTe = zeros(CVO.NumTestSets,1);
errorTr = zeros(CVO.NumTestSets,1);

%Number of trees in the forest
nTrees = 100;

%Number of features per tree
%The number of features is varied in the interval of [50, 700] 
%with step 50 and calculate for every value the CV Test error
numberOfFeatures = 450;

%Doing 5-fold CV
for fold = 1:CVO.NumTestSets
    trIdx = CVO.training(fold); %indices for training
    teIdx = CVO.test(fold);    %indices for testing
    %form testing and training sets
    tXtr = x(trIdx,:);
    ytr = y(trIdx);
    tXte = x(teIdx,:);
    yte = y(teIdx);
    
    % normalize data
    [Tr.normX, mu, sigma] = zscore(tXtr); % train, get mu and std
         
    % normalize test data
    Te.normX = normalize(tXte, mu, sigma);
            
    % Training random forest
    fprintf('Training random forest..\n');
    
    % Train the classifier: input = number of trees, train set, train labels...
    % parameters (number of features per Tree)
    classifier = TreeBagger(nTrees,Tr.normX,ytr,'Method','classification', ...
            'NVarToSample',numberOfFeatures);
  
    % Make class predictions in Test set...
    %(input = trained classifier and test set)
    predictionsTest = predict(classifier, Te.normX);
    % Make class predictions in Train set...
    %(input = trained classifier and train set)
    predictionsTrain = predict(classifier, Tr.normX);

    predictedLabelsTest = zeros(size(Te.normX,1),1);
    predictedLabelsTrain = zeros(size(Tr.normX,1),1);
    
    for i = 1:size(Te.normX,1)
        predictedLabelsTest(i) = str2num(cell2mat(predictionsTest(i)));
    end
    
    for i = 1:size(Tr.normX,1)
        predictedLabelsTrain(i) = str2num(cell2mat(predictionsTrain(i)));
    end
    
    % Tabulate the results using a confusion matrix.
    confMatTest = confusionmat(yte, categorical(predictedLabelsTest));
    confMatTrain = confusionmat(ytr, categorical(predictedLabelsTrain));
    
    % calculate BER
    berTest = BER(confMatTest);
    berTrain = BER(confMatTrain);
    
    % keep test and train error
    errorTe(fold) = berTest;
    errorTr(fold) = berTrain;
    fprintf('fold = %d, error = %.5f\n', fold, errorTe(fold));
    fprintf('fold = %d, error = %.5f\n', fold, errorTr(fold));
end