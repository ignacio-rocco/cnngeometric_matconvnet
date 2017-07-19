function [flowAff,flowAffTps,imA_warped] = computeFlow(imA,imB,netTps,netReg,netReg2,lambda) 

if nargin<6
    lambda = 0;
end
if nargin<5
    netReg2 = [];
end
if nargin<4
    netReg = [];
end

nc = @(x,L) (x-1-(L-1)/2)*2/(L-1);
uc = @(x,L) x*(L-1)/2+1+(L-1)/2;

[hA,wA]=size(imA(:,:,1));
[hB,wB]=size(imB(:,:,1));

imA_ = preprocessImage(imA);
imB_ = preprocessImage(imB);

%% estimate affine transformation
if ~isempty(netReg) && ~isempty(netReg2)
    netReg.eval({'AN1input',imA_,'AN2input',imB_});
    thetaAff1=netReg.vars(end).value;

    netReg2.eval({'AN1input',imA_,'AN2input',imB_});
    thetaAff2=netReg2.vars(end).value;

    thetaAff = (thetaAff1+thetaAff2)/2;
elseif ~isempty(netReg)
    netReg.eval({'AN1input',imA_,'AN2input',imB_});
    thetaAff=netReg.vars(end).value;
else
    thetaAff=[];
end

%% warp image with affine transformation
if ~isempty(thetaAff)
    agg = dagnn.AffineGridGenerator('Ho',hB,'Wo',wB);
    bs = dagnn.BilinearSampler; 
    grAff = agg.forward({thetaAff});
    imAaff = bs.forward({substractMeanCNN(imA),grAff{1}});  
    imAaff = substractMeanCNN(imAaff{1},1);
    imAaff_ = preprocessImage(imAaff); 
end

%% estimate TPS transformation
if ~isempty(netTps)
    if ~isempty(thetaAff)
        netTps.eval({'AN1input',imAaff_,'AN2input',imB_});
    else
        netTps.eval({'AN1input',imA_,'AN2input',imB_});
    end

    thetaNN = netTps.vars(netTps.getVarIndex('theta')).value;


    %% warp image with TPS transformation
    k=sqrt(length(thetaNN)/2);
    tgg = dagnnExtra.TpsGridGenerator('Ho',hB,'Wo',wB,'k',k,'lambda',lambda);
    bs = dagnn.BilinearSampler;  

    grTps = tgg.forward({thetaNN});
    if ~isempty(thetaAff)
        imA_warped = bs.forward({substractMeanCNN(imAaff), grTps{1}});
    else
        imA_warped = bs.forward({substractMeanCNN(imA), grTps{1}});
    end

    imA_warped = uint8(substractMeanCNN(imA_warped{1},1));  
else
    imA_warped = uint8(imAaff);
end

%% compute flow
%%%% affine only
[mgX,mgY]=meshgrid(1:wB,1:hB);
grAffY = squeeze(uc(grAff{1}(1,:,:),hA));
grAffX = squeeze(uc(grAff{1}(2,:,:),wA));
vx = grAffX - mgX;
vy = grAffY - mgY;
flowAff = cat(3,vx,vy);
    
%%%% affine + tps
if ~isempty(netTps) && ~isempty(netReg)
    % compose TPS and aff transformations
    grAffY = squeeze(uc(grAff{1}(1,:,:),hA));
    grAffX = squeeze(uc(grAff{1}(2,:,:),wA));

    % check boundaries
    badIdx = grAffY<1 | grAffY>hA | grAffX<1 | grAffX>wA;
    grAffY(badIdx)=nan;
    grAffX(badIdx)=nan;

    grTpsY = squeeze(uc(grTps{1}(1,:,:),hB));
    grTpsX = squeeze(uc(grTps{1}(2,:,:),wB));

    [mgX,mgY]=meshgrid(1:wB,1:hB);

    finalX = interp2(mgX,mgY,grAffX,grTpsX,grTpsY);
    finalY = interp2(mgX,mgY,grAffY,grTpsX,grTpsY);

    vx = finalX - mgX;
    vy = finalY - mgY;

    flowAffTps = cat(3,vx,vy);
else
    flowAffTps = [];
end
% for verification
% imB_warped2 = warp_image(single(imA),vx,vy);



