function net = createNet(varargin)

% default options
opts=struct();
opts.featureNormalization = 1;
opts.matchNormalization = 1;
opts.batchNormalization = 1;
opts.conv1size = 7;
opts.conv2size = 5;
opts.conv3size = 5;
opts.conv1filters = 128;
opts.conv2filters = 64;
opts.conv3filters = 18;
opts.featureFusionLayer = 'correlation';
opts.featExtNet = 'VGG16';
opts.featExtLastLayer = 'pool4';
opts.trainFeatExt = 0;
opts.trainFeatExtDepth = []; 
opts.trainFeatExtlrFactor = 1;
opts.geometricModel = 'affine';
opts.pathToTrainedModel = [];

% load user-specified options
opts = vl_argparse(opts, varargin);

%%%% Check if a train model should be used
if ~isempty(opts.pathToTrainedModel)
    net = load(opts.pathToTrainedModel);   
    net = net.net;
    if isa(net,'struct')
        display('converting from struct')
        net=dagnn.DagNN.loadobj(net);
    end
     
    display('using trained model');
else
%%%% Create a new model
    % Load Feature Extraction net
    if strcmp(opts.featExtNet,'VGG16')==1
        net = dagnn.DagNN.fromSimpleNN(load('imagenet-vgg-verydeep-16.mat')); 
    else
        error('Only featExtNet=''VGG16'' is implemented')
    end
    % trim FE net
    Nlayers = net.getLayerIndex(opts.featExtLastLayer);

    while length(net.layers)>Nlayers
        net.removeLayer(net.layers(end).name);
    end

    %%% duplicate AlexNet and fuse the two branches
    netStruct = net.saveobj;
    netStruct.vars(1).name='input';
    netStruct.layers(1).inputs{1}='input';

    netStruct2 = netStruct;
    % rename layers on each branch
    netStruct = netNamePrefix(netStruct,'AN1','AN1','AN');
    netStruct2 = netNamePrefix(netStruct2,'AN2','AN2','AN');

    netStructFused = fuseNetStruct(netStruct, netStruct2);

    %%% Convert back to dagNN
    net = dagnn.DagNN.loadobj(netStructFused);

    feat1Name = netStruct.vars(end).name;
    feat2Name = netStruct2.vars(end).name;
    
    if opts.featureNormalization==1 
        % add normalization layer
        net.addLayer('AN1L2featNorm', dagnn.LRN('param', [1000000 1e-6 1 0.5]), {feat1Name}, {'AN1xnorm'}, {}) ;
        net.addLayer('AN2L2featNorm', dagnn.LRN('param', [1000000 1e-6 1 0.5]), {feat2Name}, {'AN2xnorm'}, {}) ;
    	feat1Name = 'AN1xnorm';
        feat2Name = 'AN2xnorm';
    end
    
    if strcmp(opts.featureFusionLayer,'correlation')==1
        % add correlation layer
        featFusedName = 'ANxcorr';
        net.addLayer('correlate', dagnnExtra.Correlate(), {feat1Name, feat2Name}, {featFusedName}, {}) ;
    elseif strcmp(opts.featureFusionLayer,'concatenation')==1
        % add concatenation layer
        featFusedName = 'ANxconcat';
        net.addLayer('AN1concat',dagnnExtra.Concat(), {feat1Name, feat2Name}, {featFusedName}, {}) ;
    elseif strcmp(opts.featureFusionLayer,'subtract')==1
        % add subtraction layer
        featFusedName = 'ANxsubtract';
        net.addLayer('ANxsubtract',dagnnExtra.Subtract(), {feat1Name, feat2Name}, {featFusedName}, {}) ;
    end
    
    % specify size of dense features maps
    if strcmp(opts.featExtNet,'VGG16')  && strcmp(opts.featExtLastLayer,'pool4')
        fsupp_rows=15;
        fsupp_cols=15;
    else
        error('please specify the feature size for the chosen FE net and last layer')
    end
    
    % match normalization
    if opts.matchNormalization==1 
        % add normalization layer
        net.addLayer('ANcorrRelu', dagnn.ReLU(), {featFusedName}, {'ANxrelu'}, {}) ;    
        net.addLayer('ANcorrNorm', dagnn.LRN('param', [2*fsupp_rows*fsupp_cols 1e-6 1 0.5]), {'ANxrelu'}, {'ANxnorm'}, {}) ;    
        %net.addLayer('ANcorrNorm', dagnn.LRN('param', [2*fsupp_rows*fsupp_cols 1e-6 1 0.5]), {featFusedName}, {'ANxnorm'}, {}) ;    
        featFusedName = 'ANxnorm';
    end

    % add regressor net    
    net = addConvRegressionModule(net, opts, fsupp_rows, fsupp_cols, featFusedName, 'theta');

    % add loss function  
    thetaName = 'theta';
    thetaGtName = 'thetaGt';
    if strcmp(opts.geometricModel,'affine')==1
        % add affine loss layer
        lossBlock = dagnnExtra.LossAffine();
        net.addLayer('affineLoss', lossBlock, {thetaName, thetaGtName}, {'loss'}, {}) ;
    elseif strcmp(opts.geometricModel,'TPS')==1
        % as TPS parameters are already point coordinates, compare directly
        % with L2 distance
        lossBlock = dagnnExtra.LossDist2();
        net.addLayer('lossL2dist', lossBlock, {thetaName, thetaGtName}, {'loss'}, {}) ;
    end
    
end

% make list of parameters of the feature extractor which we want to
% finetune
ANparams=[];
for i=length(net.params):-1:1
    if ~isempty(opts.trainFeatExtDepth) && opts.trainFeatExt==1
        if (strcmp(net.params(i).name(1:2),'AN') && length(ANparams)<opts.trainFeatExtDepth)
            ANparams=[ANparams;i];
        end
    end
end

for i=1:length(net.params)
    % handle case without finetuning of FE
    if (strcmp(net.params(i).name(1:2),'AN') && opts.trainFeatExt==0)
        net.params(i).learningRate = 0;
    end
    % handle case with fintuning of FE
    if (strcmp(net.params(i).name(1:2),'AN') && opts.trainFeatExt==1 && ~isempty(opts.trainFeatExtdepth) && isempty(find(ANparams==i)))
        net.params(i).learningRate = 0;
    end
    % multiply feature extractor lr by factor in user parameters
    if strcmp(net.params(i).name(1:2),'AN') 
        net.params(i).learningRate = net.params(i).learningRate*opts.trainFeatExtlrFactor;
    end   
    display(['P' num2str(i) ': ' net.params(i).name ', ' net.params(i).trainMethod ', lr: ' num2str(net.params(i).learningRate)]);
end

net.conserveMemory = true; % false: store all the intermediate results
net.vars(end).precious=1;
