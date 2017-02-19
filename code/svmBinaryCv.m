%% SVM binary classifier
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

%Setting the regularization parameter C
%Setting the regularization parameter C
%C is varied in the interval of [10^(-6), 1] with 5 points in between
%and calculate for every C the CV Test error
C = 1e-3;

%Doing 5-fold CV
for fold = 1:CVO.NumTestSets
    trIdx = CVO.training(fold); %indices for training
    teIdx = CVO.test(fold);    %indices for testingj,vb ─µ 
    %form testing and training sets
    tXtr = x(trIdx,:);
    ytr = y(trIdx);
    tXte = x(teIdx,:);
    yte = y(teIdx);
    
    % normalize data
    [Tr.normX, mu, sigma] = zscore(tXtr); % train, get mu and std
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%UNCOMMENT THE FOLLOWING BLOCK OF CODE IF YOU WANT TO PERFORM PCA%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %fprintf('Using PCA on training data...\n');
    %Use PCA to select most important coef
    %[coefs,scores,variances] = pca(Tr.normX);
    
    %choose the dimension d to keep
    %We experimented with different values of d
    %and perform CV 
    
    %percent_explained = 100 * variances / sum(variances);
    %pervar = 100*cumsum(variances) / sum(variances);
    
    %CHOOSE THE NUMBER OF PRINCIPAL COMPONENTS
    %d = max(find(pervar < 95));
    %d = 400;
    %new variables in pca subspace
    %new_variables = Tr.normX*coefs(:,1:d);
    %Tr.normX = new_variables;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % normalize test data
    Te.normX = normalize(tXte, mu, sigma);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%UNCOMMENT THE FOLLOWING BLOCK OF CODE IF YOU WANT TO PERFORM PCA%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % project in pca subspace
    %fprintf('Projecting test data into PCA subspace...\n');
    %testScores = Te.normX*coefs(:,1:d);
    %Te.normX = testScores;
    %fprintf('End of PCA\n');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Training SVM
    fprintf('Training SVM..\n');

    % setting svm parameters (C and type of kernel)
    t = templateSVM('BoxConstraint',C,'KernelFunction','linear');
    % Train the classifier: input = train set, train labels, parameters
    classifier = fitcecoc(Tr.normX, ytr, 'Learners', t);
    
    % Make class predictions in Test set...
    %(input = trained classifier and test set)
    predictedLabelsTest = predict(classifier, Te.normX);
    % Make class predictions in Train set...
    %(input = trained classifier and train set)
    predictedLabelsTrain = predict(classifier, Tr.normX);

    % Tabulate the results using a confusion matrix.
    confMatTest = confusionmat(yte, predictedLabelsTest);
    confMatTrain = confusionmat(ytr, predictedLabelsTrain);
    
    % calculate BER
    berTest = BER(confMatTest);
    berTrain = BER(confMatTrain);
    
    % keep test and train error
    errorTe(fold) = berTest;
    errorTr(fold) = berTrain;
    fprintf('fold = %d, error = %.5f\n', fold, errorTe(fold));
    fprintf('fold = %d, error = %.5f\n', fold, errorTr(fold));
end