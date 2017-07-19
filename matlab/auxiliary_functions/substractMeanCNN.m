function im1 = substractMeanAlexNet(im1,reverse)

if nargin<2
    reverse=0;
else
    reverse=1;
end

averageImage=[122.7395;114.8967;101.5919];
im1 = single(im1) ; % note: 0-255 range
%im1_ = imresize(im1_, imageSize(1:2)) ;
if reverse==0
    im1 = im1 - repmat(repmat(permute(averageImage,[3 2 1]),size(im1,1),size(im1,2)),[1,1,1,size(im1,4)]);
else
    im1 = im1 + repmat(repmat(permute(averageImage,[3 2 1]),size(im1,1),size(im1,2)),[1,1,1,size(im1,4)]);
end