function inputs = getDagNNBatch(topts,imagesA,theta,batchIdx)

% specify batch image size (CNN input size)
imsize = [227 227];

% get images and transformation parameters from the specified batchIdx
theta_batch = single(theta(:,:,:,batchIdx));
%im_raw = subMeanAndReshape(vl_imreadjpeg(imagesA(batchIdx))); % images in imagesA should be of the same resolution
im_raw = subMeanAndReshape(vl_imreadjpeg(imagesA(batchIdx),'resize',[640 480])); % images in imagesA should be of the same resolution
[H,W]=size(im_raw(:,:,1,1));

% generate image A by cropping the raw image
innerCropfactor = 9/16;
imA_batch = im_raw(round(H*(1-innerCropfactor)/2+1):end-round(H*(1-innerCropfactor)/2),round(W*(1-innerCropfactor)/2+1):end-round(W*(1-innerCropfactor)/2),:,:);
% resize to the batch image size
imA_batch = imresize(imA_batch,imsize);

% add extra padding for enlarging the sampling region for image B
paddingFactor = 1/2;
im_raw = imresize(im_raw,[454 454]);  % delete line
im_raw=padarray(im_raw, size(im_raw(:,:,1,1))*paddingFactor, 'symmetric');    
factor = paddingFactor*innerCropfactor;        

% generate image B by transforming image A
if strcmp(topts.geometricModel,'affine')==1        
    % use affine transformation
    tnf = dagnn.AffineGridGenerator('Ho',imsize(1),'Wo',imsize(2)); 
elseif strcmp(topts.geometricModel,'TPS')==1
    % use TPS transformation
    tnf = dagnnExtra.TpsGridGenerator('Ho',imsize(1),'Wo',imsize(2));        
end
samplingGrid = tnf.forward({theta_batch});
bs = dagnn.BilinearSampler;
imB_batch = bs.forward({im_raw,samplingGrid{1}*factor});
imB_batch = imB_batch{1};

% copy to GPU memory if needed
if ~isempty(topts.gpus)
    imA_batch = gpuArray(imA_batch) ;
    imB_batch = gpuArray(imB_batch) ;
    theta_batch =  gpuArray(theta_batch) ;
end

% return batch data
inputs = {'thetaGt', theta_batch, 'AN1input', imA_batch, 'AN2input', imB_batch};