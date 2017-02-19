%% SVM multiclass classifier
%% MAKE PREDICTIONS TO GIVEN TEST SET
% Giannakopoulos Athanasios
% Kyritsis Georgios

clearvars; close all; clear;

% Load features and labels of training data
load train.mat;

% Load Test Set
load test.mat;
xTest = test.X_cnn;


%add path for normalize function
addpath(genpath('/home/kyritsis/myfiles/Machine/code/DeepLearnToolbox-master'));

%Extracting the CNN features and the corresponding labels
%Change to train.X_hog if you want to use HOG features
x = train.X_cnn;
y = train.y;

%set seed
rng(8339);

%Setting the regularization parameter C
C = 1e-3;

% normalize data
[Tr.normX, mu, sigma] = zscore(x); % train, get mu and std
    
fprintf('Using PCA on training data...\n');
%Use PCA to select most important coef
[coefs,scores,variances] = pca(Tr.normX);
    
%CHOOSE THE NUMBER OF PRINCIPAL COMPONENTS
d = 400;
%new variables in pca subspace
new_variables = Tr.normX*coefs(:,1:d);
Tr.normX = new_variables;
    
% normalize test data
Te.normX = normalize(xTest, mu, sigma);
    
% project in pca subspace
fprintf('Projecting test data into PCA subspace...\n');
testScores = Te.normX*coefs(:,1:d);
Te.normX = testScores;
fprintf('End of PCA\n');

% Training SVM
fprintf('Training SVM..\n');

% setting svm parameters (C and type of kernel)
t = templateSVM('BoxConstraint',C,'KernelFunction','linear');
% Train the classifier: input = train set, train labels, parameters
classifier = fitcecoc(Tr.normX, y, 'Learners', t);

% Make class predictions in Test set...
%(input = trained classifier and test set)
Ytest = predict(classifier, Te.normX);

% assign your predicted scores to Ytest first, then:
save('pred_multiclass', 'Ytest');