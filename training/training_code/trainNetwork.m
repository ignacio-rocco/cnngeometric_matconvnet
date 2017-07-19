function result=trainNetwork(paths, usertopts, mode)
% path: struct specifying location of data and results
% usertopts: struct with training options provided by user 
% mode: 1 train model  (will initialize new model)
%       2 evaluate     (will load existing model)

%% Define default training parameters
topts=struct();
topts.trSet=[];
topts.resDir=[];
topts.lr=1e-3;
topts.numEpochs = 10;
topts.batchSize=16;
topts.gpus=1;
topts.saveResults=1;
topts.weightDecay = 0 ;
topts.momentum = 0.9 ;

% define default network parameters
topts.featureNormalization = 1;
topts.matchNormalization = 1;
topts.batchNormalization = 1;
topts.conv1size = 7;
topts.conv2size = 5;
topts.conv3size = 5;
topts.conv1filters = 128;
topts.conv2filters = 64;
topts.conv3filters = 18;
topts.featureFusionLayer = 'correlation';
topts.featExtNet = 'VGG16';
topts.featExtLastLayer = 'pool4';
topts.trainFeatExt = 0;
topts.trainFeatExtDepth = []; 
topts.trainFeatExtlrFactor = 1;
topts.geometricModel = 'affine';
topts.pathToTrainedModel = [];

%% Load user training options

% overwrite default parameters with user provided parameters (if any)
if nargin>=2
    topts = vl_argparse(topts, usertopts);
end

% set default mode (if not specified by user)
if nargin<3
    mode=1;
end

% verify required parameters
if isempty(topts.trSet) || isempty(topts.resDir) 
    error('you need to specify: topts.trSet, topts.resDir');
end

display('Training options:');
topts

%% create result directory for this experiment
resDirFullPath =  fullfile(paths.results,topts.resDir,'/');

if mode==1
    [~, ~] = system(['mkdir ' paths.results]);
    [~, ~] = system(['mkdir ' resDirFullPath]);
end

%% Prepare net
% set random seed 
seed=1737;
rng(seed)

% remove options related to optimization and keep options related to net 
netopts=rmfield(topts,{'trSet','resDir','lr','numEpochs','batchSize','gpus','saveResults','weightDecay','momentum'});

% create or load trained net
if mode==1 || mode==3 % training mode: initialize new model
    display('Creating NET');
    net = createNet(netopts);
else       % evaluation mode: load model
    display('Loading trained NET model');
    load([resDirFullPath 'net.mat']);  
end

% save network diagram and training options (in training mode only)
if mode==1
    plotNet(net,'plotTexts',1,'figNum',1500,'textSize',3,'spacingHLayers',8,'spacingHInputs',25);
    print([resDirFullPath 'net-diagram.pdf'],'-dpdf')
    %snapshot([resDirFullPath 'net-model.pdf'],9);
    save([resDirFullPath 'topts.mat'], 'topts');
    save([resDirFullPath 'netopts.mat'], 'netopts');
end

%% Load training data
if topts.trSet(1)=='/' % if absolute path is specified
    dataFile = fullfile(topts.trSet,'trainValData.mat')
else                   % if no absolute path is specified, use relative path
    dataFile = fullfile(paths.baseDir,'training','training_data',topts.trSet,'trainValData.mat')
end

trValData = load(dataFile);

% concat tr and val data (for passing to training algorithm)
Ntr = length(trValData.tr.imageA);
Nval = length(trValData.val.imageA);

imagesA=[strcat(paths.trValdatasetPath,trValData.tr.imageA);... % prepend path to dataset
         strcat(paths.trValdatasetPath,trValData.val.imageA)];
theta=cat(4,trValData.tr.theta,trValData.val.theta);

%% Train 

% define options for the training algorithm
opts=struct();
% copy parameters from topts
opts.weightDecay = topts.weightDecay;
opts.momentum = topts.momentum;
opts.numEpochs = topts.numEpochs;
opts.batchSize = topts.batchSize;
opts.gpus = topts.gpus;
opts.expDir = resDirFullPath;
% define tr/val split indices
opts.train = 1:Ntr;
opts.val = Ntr+1:(Ntr+Nval);
% define lr decay scheme as k/sqrt(t) or use user defined decay scheme
if length(topts.lr)==1    
    opts.learningRate = topts.lr./sqrt([1:opts.numEpochs]);  
else
    opts.learningRate = topts.lr;
end
% define loss variable, with derivative=1, where backpropagation is
% started from
opts.derOutputs = {'loss', 1};    

% define function to get a training batch
getBatch = @(batchIdx) getDagNNBatch(topts,imagesA,theta,batchIdx);

stats=[];

if mode==1
    % train
    [net,stats]=cnn_train_dag_custom(net, getBatch, opts);
end

%% Return struct with useful stuff for evaluation
result.imagesA = imagesA;
result.theta = theta;
result.opts = opts;
result.topts = topts;
result.net = net;
result.stats = stats;
result.getBatch = getBatch;

