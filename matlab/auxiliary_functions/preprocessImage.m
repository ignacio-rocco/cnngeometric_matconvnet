function im1_ = preprocessImage(im1,imageSize)
if nargin<2
    imageSize=[227 227 3];
end
averageImage=[122.7395;114.8967;101.5919];
im1_ = single(im1) ; % note: 0-255 range
im1_ = imresize(im1_, imageSize(1:2)) ;
im1_ = im1_ - repmat(permute(averageImage,[3 2 1]),imageSize(1),imageSize(2));
