function results = evaluatePropFlowPascal(paths,varargin)

% Proposal Flow dataset base path
pfPascalPath = paths.pfPascalPath;

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
    lambda=0;
else
    doTps = 0;
end

%% Load dataset pairs
load(fullfile(pfPascalPath,'test_pairs_pf_pascal.mat')) ; % load pair list
Npairs = length(pairs.cat);

%% Auxiliary functions 
normcoords = @(x,L) (x-1-(L-1)/2)*2/(L-1);
unormcoords = @(x,L) x*(L-1)/2+1+(L-1)/2;

%% compute PCK
pckVec = zeros(Npairs,1);

if doAffine==1
    thetaAffVec = zeros(1,1,6,Npairs,'single');
else
    thetaAffVec=[];
end
if doTps==1
    thetaSize = size(netTps.params(netTps.getParamIndex(netTps.layers(netTps.getLayerIndex('conv3')).params{1})).value,4);
    k = sqrt(thetaSize/2);
    lambda = 0;
    thetaTpsVec = zeros(1,1,thetaSize,Npairs,'single');
else
    thetaTpsVec=[];
end

for idx=1:Npairs
    %%%% load images
    imA = imread(fullfile(pfPascalPath, 'JPEGImages', pairs.imageA{idx}));
    imB = imread(fullfile(pfPascalPath, 'JPEGImages', pairs.imageB{idx}));
    
    [hA,wA]=size(imA(:,:,1));
    [hB,wB]=size(imB(:,:,1));
    
    %%%% load points
    XA = pairs.XA{idx};
    YA = pairs.YA{idx};
    XB = pairs.XB{idx};
    YB = pairs.YB{idx};
    
    %%%% normalize points
    XB_ = normcoords(XB,wB);
    YB_ = normcoords(YB,hB);
        
    %%%% define reference length for pck computation
    Lpck = max(max(XA)-min(XA),max(YA)-min(YA));    

    %%%% normalize images for CNN
    imA_ = preprocessImage(imA);
    imB_ = preprocessImage(imB);
   
    if useGPU
        imA_ = gpuArray(imA_);
        imB_ = gpuArray(imB_);
    end
    
    %%%% compute affine model paramters
    if doAffine == 1
        netReg.eval({'AN1input',imA_,'AN2input',imB_});
        thetaAff1=netReg.vars(end).value;

        if evalopts.ensemble==1
            netReg2.eval({'AN1input',imA_,'AN2input',imB_});
            thetaAff2=netReg2.vars(end).value;

            thetaAff = (thetaAff1+thetaAff2)/2;
        else
            thetaAff = thetaAff1;
        end  

        if length(thetaAff)==4
            thetaAff = cat(3,thetaAff(1),0,0,thetaAff(2:end));
        end

        thetaAffVec(:,:,:,idx)=gather(thetaAff);
    else
        thetaAff = permute(single([1 0 0 1 0 0]),[1 3 2]);
    end

    %%%% warp image using affine model
    if doAffine==1
        afs = dagnn.AffineGridGenerator('Ho',227,'Wo',227);
        bs = dagnn.BilinearSampler; 
        grAff = afs.forward({thetaAff});
        imAaff_ = bs.forward({imA_,grAff{1}});  imAaff_ = imAaff_{1};
    else
        imAaff_ = imA_;
    end
        
    % compute TPS model paramters
    if doTps==1
        netTps.eval({'AN1input',imAaff_,'AN2input',imB_});
        thetaTps = netTps.vars(netTps.getVarIndex('theta')).value;

        thetaTpsVec(:,:,:,idx)=gather(thetaTps);
    end

    %%%% transform points 
    if doTps==1 && doAffine==1
        % reverse TPS transformation
        tpsTransfPoints = dagnnExtra.TpsTransform('k',k,'lambda',lambda);
        ptsBtps_ = tpsTransfPoints.forward({thetaTps,{[XB_ YB_]}});
        ptsBtps_ = ptsBtps_{1}{1};    
        % reverse affine transformation
        ptsBtpsaffX_ = thetaAff(4)*ptsBtps_(:,1)+thetaAff(3)*ptsBtps_(:,2)+thetaAff(6);    
        ptsBtpsaffY_ = thetaAff(2)*ptsBtps_(:,1)+thetaAff(1)*ptsBtps_(:,2)+thetaAff(5);    

        ptsBtpsaffX = unormcoords(ptsBtpsaffX_,wA);
        ptsBtpsaffY = unormcoords(ptsBtpsaffY_,hA);

   elseif doTps==1 && doAffine==0
         % reverse TPS transformation
        tpsTransfPoints = dagnnExtra.TpsTransform('k',k,'lambda',lambda);
        ptsBtps_ = tpsTransfPoints.forward({thetaTps,{[XB_ YB_]}});
        ptsBtps_ = ptsBtps_{1}{1};    
        % dont do affine transformation
        ptsBtpsaffX = unormcoords(ptsBtps_(:,1),wA);
        ptsBtpsaffY = unormcoords(ptsBtps_(:,2),hA);
    elseif doTps==0 && doAffine==1
        % reverse only affine transformation
        ptsBtpsaffX_ = thetaAff(4)*XB_+thetaAff(3)*YB_+thetaAff(6);    
        ptsBtpsaffY_ = thetaAff(2)*XB_+thetaAff(1)*YB_+thetaAff(5);    

        ptsBtpsaffX = unormcoords(ptsBtpsaffX_,wA);
        ptsBtpsaffY = unormcoords(ptsBtpsaffY_,hA);        
    else
        error('You need to use at least one of tps/aff')
    end
        
        
    if false
        colors = hsv(length(XA));
        % Keypoint distance before alignment
        figure(1); clf; imshow(imA); hold on
        scatter(XA,YA,500,colors,'.');
        scatter(unormcoords(XB_,wA),unormcoords(YB_,hA),500,colors,'.');  
        plot([XA unormcoords(XB_,wA)]',[YA unormcoords(YB_,hA)]','Color','k');
        
        % Keypoint distance after alignment
        figure(2); clf; imshow(imA); hold on
        scatter(XA,YA,500,colors,'.');
        scatter(ptsBtpsaffX,ptsBtpsaffY,50,colors,'o');   
        plot([XA ptsBtpsaffX]',[YA ptsBtpsaffY]','Color','k');
    end
    
    % compute PCK
    dist = gather(sqrt((squeeze(ptsBtpsaffX)-XA).^2+(squeeze(ptsBtpsaffY)-YA).^2));
    alpha = 0.1;
    pckVec(idx) = sum(dist<=alpha*Lpck)/length(dist);
    
    display([num2str(idx) '/' num2str(Npairs)])
end

meanPck = mean(pckVec);

% make struct with results
results = struct();
results.evalopts = evalopts;
results.thetaAffVec = thetaAffVec;
results.thetaTpsVec = thetaTpsVec;
results.pckVec = pckVec;
results.meanPck = meanPck;


