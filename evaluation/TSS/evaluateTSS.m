function results = evaluateTSS(paths,varargin)

% TSS dataset base path
datasetDir = paths.TSSPath;
% TSS results path
resultsDir = fullfile(paths.results,'evalTSS');
[~,~]=system(['mkdir "' paths.results '"']);
[~,~]=system(['mkdir "' resultsDir '"']);

% default parameters
evalopts.ensemble=0;
evalopts.netmodelpath='';
evalopts.tpsnet = [];
evalopts.affnet = [];
evalopts.affnet2 = [];
evalopts.useGPU=0;

% override with user provided parameters
evalopts = vl_argparse(evalopts,varargin);

% check if affine ensemble is specified 
if ~isempty(evalopts.affnet) && ~isempty(evalopts.affnet2)
    evalopts.ensemble=1;
else
    evalopts.ensemble=0;
end

useGPU = evalopts.useGPU;

%% load trained models

netTps=[];
netReg=[];
netReg2=[];

% load affine CNN
if ~isempty(evalopts.affnet)
    doAffine = 1;
    try
        netReg=load(fullfile(evalopts.netmodelpath,evalopts.affnet,'net.mat'));     
        netReg=netReg.net;
    catch
        netReg=load(fullfile(evalopts.netmodelpath,evalopts.affnet,'net-epoch-10.mat'));
        netReg=dagnn.DagNN.loadobj(netReg.net);
    end
    netReg.removeLayer(netReg.layers(end).name);
    netReg.vars(end).precious=1;
    netReg.mode = 'test';
    if useGPU; netReg.move('gpu'); end
else
    doAffine = 0;
end

if evalopts.ensemble==1 
    try
        netReg2=load(fullfile(evalopts.netmodelpath,evalopts.affnet2,'net.mat'));     
        netReg2=netReg2.net;
    catch
        netReg2=load(fullfile(evalopts.netmodelpath,evalopts.affnet2,'net-epoch-10.mat'));
        netReg2=dagnn.DagNN.loadobj(netReg2.net);
    end
    netReg2.removeLayer(netReg2.layers(end).name);
    netReg2.vars(end).precious=1;
    netReg2.mode = 'test';
    if useGPU; netReg2.move('gpu'); end
end

% load TPS CNN
if ~isempty(evalopts.tpsnet)
    doTps = 1;
    try
        netTps=load(fullfile(evalopts.netmodelpath,evalopts.tpsnet,'net.mat'));     
        netTps=netTps.net;
    catch
        netTps=load(fullfile(evalopts.netmodelpath,evalopts.tpsnet,'net-epoch-10.mat'));
        netTps=dagnn.DagNN.loadobj(netTps.net);
    end
    netTps.vars(netTps.getVarIndex('theta')).precious=1;
    netTps.removeLayer(netTps.layers(end).name);
    netTps.mode = 'test';
    if useGPU; netTps.move('gpu'); end
    lambda=0
else
    doTps = 0;
end

%% normalization functions
nc = @(x,L) (x-1-(L-1)/2)*2/(L-1);
uc = @(x,L) x*(L-1)/2+1+(L-1)/2;

%% load test data folders and make result dirs

keepDir = @(x) x([x.isdir]);
removeDotDirs = @(y) y(arrayfun(@(x) x.name(1), y) ~= '.');
getDir = @(x) removeDotDirs(keepDir(dir(x)));

categories = getDir(datasetDir); categories = {categories.name};

testData={};

for i=1:length(categories)
    imDirs = getDir(fullfile(datasetDir,categories{i})); imDirs = {imDirs.name};
    imDirs = strcat([categories{i} '/'],imDirs);
    testData = [testData imDirs];
    [~,~]=system(['mkdir "' fullfile(resultsDir,categories{i}) '"']);
end

%% compute Flow results
for i=1:length(testData)
    [~,~]=system(['mkdir "' fullfile(resultsDir,testData{i}) '"']);
    % load images
    flip = importdata(fullfile(datasetDir,testData{i},'flip_gt.txt'));
    im1 = imread(fullfile(datasetDir,testData{i},'image1.png'));
    im2 = imread(fullfile(datasetDir,testData{i},'image2.png'));
    
    % compute flow 1 (warp im2 into im1)
    if flip==1
        im2prime = im2(:,end:-1:1,:);
    else
        im2prime = im2;
    end
    
    [flowAff1,flowAffTps1,im2_warped] = computeFlow(im2prime,im1,netTps,netReg,netReg2);
    
    % compute flow 2 (warp im1 into im2)
    if flip==1
        im1prime = im1(:,end:-1:1,:);
    else
        im1prime = im1;
    end
    [flowAff2,flowAffTps2,im1_warped] = computeFlow(im1prime,im2,netTps,netReg);
    
    if ~isempty(netTps)
        writeFlowFile(flowAffTps1,fullfile(resultsDir,testData{i},'flow1.flo'));
        writeFlowFile(flowAffTps2,fullfile(resultsDir,testData{i},'flow2.flo'));
    else
        writeFlowFile(flowAff1,fullfile(resultsDir,testData{i},'flow1.flo'));
        writeFlowFile(flowAff2,fullfile(resultsDir,testData{i},'flow2.flo'));
    end
    display([num2str(i),'/',num2str(length(testData))]);
end

%% compute evaluation using TSS Evaluation toolkit
cd(fullfile(paths.baseDir,'evaluation','TSS','TSS_CVPR2016_EvaluationKit-master','EvalToolMATLAB'));

RunEvaluation(fullfile(resultsDir,'FG3DCar'),fullfile(datasetDir,'FG3DCar'));
RunEvaluation(fullfile(resultsDir,'PASCAL'),fullfile(datasetDir,'PASCAL'));
RunEvaluation(fullfile(resultsDir,'JODS'),fullfile(datasetDir,'JODS'));

results.pckFG3DCar = importdata(fullfile(resultsDir,'FG3DCar','scores.csv'));
results.pckPASCAL = importdata(fullfile(resultsDir,'PASCAL','scores.csv'));
results.pckJODS = importdata(fullfile(resultsDir,'JODS','scores.csv'));

results.meanPckFG3DCar = mean(results.pckFG3DCar.data(1:end-2,7));
results.meanPckPASCAL = mean(results.pckPASCAL.data(1:end-2,7));
results.meanPckJODS = mean(results.pckJODS.data(1:end-2,7));

results.meanPckAll = mean([results.pckFG3DCar.data(1:end-2,7);results.pckPASCAL.data(1:end-2,7);results.pckJODS.data(1:end-2,7)]);





    
