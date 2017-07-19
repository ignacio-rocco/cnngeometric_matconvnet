function results = evaluateCaltech(paths,varargin)

% Caltech dataset base path
caltech101Path = paths.caltech101Path

% default parameters
evalopts.ensemble=0;
evalopts.netmodelpath='';
evalopts.tpsnet = 'tps';
evalopts.affnet = 'aff';
evalopts.affnet2 = [];
evalopts.useGPU=0;

% override with user provided parameters
evalopts = vl_argparse(evalopts,varargin);

% check if ensemble is specified and adjust resultname
if ~isempty(evalopts.affnet) && ~isempty(evalopts.affnet2)
    evalopts.ensemble=1;
    evalopts.resultname='ensemble_eval.mat';
else
    evalopts.ensemble=0;
end

useGPU = evalopts.useGPU;

%% load trained models
% load affine CNN
if ~isempty(evalopts.affnet)
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
    netReg=[];
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
else
    netReg2=[];
end

% load TPS CNN
if ~isempty(evalopts.tpsnet)
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
    netTps = [];
end

%% Load dataset pairs
load(fullfile(caltech101Path,'test_pairs_caltech.mat')) ; % load pair list
Npairs = length(pairs.imagesA);

%% Compute evaluation for the whole dataset

% allocate space for storing results
IoUaff=zeros(Npairs,1);
IoUafftps=zeros(Npairs,1);
LTACCaff=zeros(Npairs,1);
LTACCafftps=zeros(Npairs,1);
LOCERRaff=zeros(Npairs,1);
LOCERRafftps=zeros(Npairs,1);

validPairs = ones(Npairs,1);

hf=[]

% check for poly2mask function availability
if exist('poly2mask')==2
    p2m = @(X,Y,M,N) poly2mask(X,Y,M,N); % use MATLAB's fn
else
    p2m = @(X,Y,M,N) poly2mask_octave(X,Y,M,N); % use provided fn in ./matlab/auxfn
end

for idx=1:Npairs
    % Load images
    imA = imread(fullfile(caltech101Path,pairs.imagesA{idx}));
    if size(imA,3)==1
        imA=repmat(imA,1,1,3);
    end
    imB = imread(fullfile(caltech101Path,pairs.imagesB{idx}));
    if size(imB,3)==1
        imB=repmat(imB,1,1,3);
    end
    % Load annotation files     
	annA = load(fullfile(caltech101Path,pairs.annotA{idx}));    
    annB = load(fullfile(caltech101Path,pairs.annotB{idx}));    
    
    % check if annotation contains at least 3 vertices
    if size(annA.obj_contour,2)<3 || size(annB.obj_contour,2)<3
       validPairs(idx)=0;
       continue;
    end
    
    % Extract mask images from polygonal annotations    
    mskA = p2m(annA.obj_contour(1,:)+annA.box_coord(3), annA.obj_contour(2,:)+annA.box_coord(1), size(imA,1), size(imA,2));
    mskB = p2m(annB.obj_contour(1,:)+annB.box_coord(3), annB.obj_contour(2,:)+annB.box_coord(1), size(imB,1), size(imB,2));
    
    % check if annotation area is insignificant (<1%)    
    if length(find(mskA(:)~=0))/length(mskA(:))<0.01 || length(find(mskB(:)~=0))/length(mskB(:))<0.01
        validPairs(idx)=0;
        continue;
    end
    
    [hA,wA]=size(imA(:,:,1));
    [hB,wB]=size(imB(:,:,1));
    
    [flowAff,flowAffTps] = computeFlow(imA,imB,netTps,netReg,netReg2,0);
    
    try % compute metrics
        [seg, accuracyAff] = TransferLabelAndEvaluateAccuracy(mskB, mskA, round(flowAff(:,:,1)), round(flowAff(:,:,2)));
        IoUaff(idx)=accuracyAff.iou;
        LTACCaff(idx)=accuracyAff.mean;
        LOCERRaff(idx)=accuracyAff.loc_err;   
        
        [seg, accuracyAffTps] = TransferLabelAndEvaluateAccuracy(mskB, mskA, round(flowAffTps(:,:,1)), round(flowAffTps(:,:,2)));
        IoUafftps(idx)=accuracyAffTps.iou;
        LTACCafftps(idx)=accuracyAffTps.mean;
        LOCERRafftps(idx)=accuracyAffTps.loc_err;
    catch
        validPairs(idx)=0;
    end
    
    display([num2str(idx) '/' num2str(Npairs)])
end

% results
results=struct();
results.IoUaff=IoUaff;
results.IoUafftps=IoUafftps;
results.LTACCaff=LTACCaff;
results.LTACCafftps=LTACCafftps;
results.LOCERRaff=LOCERRaff;
results.LOCERRafftps=LOCERRafftps;
results.validPairs = validPairs; % detected invalid idx: 364, 1497

