%% NN multiclass classifier
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

%Converting the labels to categorical
y = categorical(y);

%fix seed
rng(8339);

%Create a random partition for 5-fold cross-validation
%y is a categorical variable 
%Each subsample has roughly the same size 
%and roughly the same class proportions as in y
CVO = cvpartition(y,'KFold',5);

%Initializing vector for holding the Test errors for each k-fold
errorTe = zeros(CVO.NumTestSets,1);

%Setting number of units in hidden layer
%The number of units is varied in the interval of [10, 50] 
%with step 10 and calculate for every value the CV Test error
units = 30;

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
    
    %Create neural network 
    nn = nnsetup([size(Tr.normX,2) units 4]);
    
    opts.numepochs =  40;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples
    % if == 1 => plots training error as the NN is trained
    opts.plot = 0;
    nn.learningRate = 2;
    
    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.normX) / opts.batchsize);
    Tr.normX = Tr.normX(1:numSampToUse,:);
    ytr = ytr(1:numSampToUse);
    
    % normalize data
    % prepare labels for NN
    LL = [1*(double(ytr) == 1), 1*(double(ytr) == 2), 1*(double(ytr) == 3), 1*(double(ytr) == 4) ];
    [nn, L] = nntrain(nn, Tr.normX, LL, opts);
    Te.normX = normalize(tXte, mu, sigma);  % normalize test data
    
    % to get the scores we need to do nnff (feed-forward)
    % see for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
    nn.testing = 0;
    
    % predict on the test set
    nnPred = nn.a{end};
    % get the most likely class
    [~,classVote] = max(nnPred,[],2);
    
    testLabels = double(y(teIdx));
    predictedLabels = classVote;
    
    % Tabulate the results using a confusion matrix.
    confMat = confusionmat(testLabels, predictedLabels);
    
    % calculate BER
    ber = BER(confMat);
    
    % keep test error
    errorTe(fold) = ber;    
    fprintf('fold = %d, error = %.5f\n', fold, errorTe(fold));
end