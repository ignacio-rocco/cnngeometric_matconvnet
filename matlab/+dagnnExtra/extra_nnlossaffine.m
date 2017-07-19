function [output] = extra_nnlossaffine(thetaNn,thetaGt,dzdy,numderiv)

if nargin<4
    numderiv=0;
end
if nargin<3
    dzdy=[];
end

delta = 1e-3; % perturbation for numerical gradient computation

% define grid
[X,Y]=meshgrid(-1:0.1:1,-1:0.1:1);

if length(size(thetaNn))==4
    batchSize = size(thetaNn,4);
else 
    batchSize=1;
end

if size(thetaNn,3)==4
    thetaNn6=zeros(1,1,6,batchSize,'single');
    if ~isa(thetaNn,'single') 
        thetaNn6 = gpuArray(thetaNn6);
    end
    thetaNn6(:,:,[1 4 5 6],:)=thetaNn; %scy scx ty tx
    thetaNn=thetaNn6;
    similarity=1;
else
    similarity=0;
end


% replicate the theta vectors in dim1 and dim2 to be able to compute
% element-wise product
blockSize = size(X);
thetaNnBlock = repmat(thetaNn,[blockSize(1),blockSize(2),1,1]);
thetaGtBlock = repmat(thetaGt,[blockSize(1),blockSize(2),1,1]);

% replicate the X and Y matrices in dim4 to be able to compute element-wise
% product
XBlock = repmat(X,[1 1 1 batchSize]);
YBlock = repmat(Y,[1 1 1 batchSize]);

if ~isa(thetaNn,'single')
    XBlock = gpuArray(XBlock);
    YBlock = gpuArray(YBlock);
end

if nargin < 3 || isempty(dzdy) % forward pass, output=loss
    
    X_A_NN = thetaNnBlock(:,:,4,:).*XBlock + thetaNnBlock(:,:,3,:).*YBlock + thetaNnBlock(:,:,6,:);
    Y_A_NN = thetaNnBlock(:,:,2,:).*XBlock + thetaNnBlock(:,:,1,:).*YBlock + thetaNnBlock(:,:,5,:);

    X_A_GT = thetaGtBlock(:,:,4,:).*XBlock + thetaGtBlock(:,:,3,:).*YBlock + thetaGtBlock(:,:,6,:);
    Y_A_GT = thetaGtBlock(:,:,2,:).*XBlock + thetaGtBlock(:,:,1,:).*YBlock + thetaGtBlock(:,:,5,:);

%     % rescale for loss
%     X_A_NN = (X_A_NN+1)*(w-1)/2+1; Y_A_NN = (Y_A_NN+1)*(h-1)/2+1;        
%     X_A_GT = (X_A_GT+1)*(w-1)/2+1; Y_A_GT = (Y_A_GT+1)*(h-1)/2+1;

    %loss = sum(sum(sqrt((X_A_NN-X_A_GT).^2+(Y_A_NN-Y_A_GT).^2),1),2)/prod(blockSize); 
    loss = sum(sum((X_A_NN-X_A_GT).^2+(Y_A_NN-Y_A_GT).^2,1),2)/prod(blockSize); 
    %loss = sum(sum((X_A_NN-X_A_GT).^2+(Y_A_NN-Y_A_GT).^2,1),2); % real
    % loss used for derivatives

    %output= mean(squeeze(loss)); % return mean loss across the batch samples
    % PREVIOUSLY: 
    %output= sum(squeeze(loss)); % return mean loss across the batch samples
    output = squeeze(loss); % return mean loss across the batch samples
        
else % backward pass, output=derivative            
    if numderiv==0       
        X_A_NN = thetaNnBlock(:,:,4,:).*XBlock + thetaNnBlock(:,:,3,:).*YBlock + thetaNnBlock(:,:,6,:);
        Y_A_NN = thetaNnBlock(:,:,2,:).*XBlock + thetaNnBlock(:,:,1,:).*YBlock + thetaNnBlock(:,:,5,:);

        X_A_GT = thetaGtBlock(:,:,4,:).*XBlock + thetaGtBlock(:,:,3,:).*YBlock + thetaGtBlock(:,:,6,:);
        Y_A_GT = thetaGtBlock(:,:,2,:).*XBlock + thetaGtBlock(:,:,1,:).*YBlock + thetaGtBlock(:,:,5,:);

        dldt4 = 2*sum(sum((X_A_NN-X_A_GT).*XBlock,1),2);
        dldt1 = 2*sum(sum((Y_A_NN-Y_A_GT).*YBlock,1),2);
        dldt3 = 2*sum(sum((X_A_NN-X_A_GT).*YBlock,1),2);
        dldt2 = 2*sum(sum((Y_A_NN-Y_A_GT).*XBlock,1),2);
        dldt6 = 2*sum(sum((X_A_NN-X_A_GT),1),2);
        dldt5 = 2*sum(sum((Y_A_NN-Y_A_GT),1),2);

        %output = cat(3,dldt1,dldt2,dldt3,dldt4,dldt5,dldt6)./prod(blockSize)./batchSize;
        if similarity==0
            output = cat(3,dldt1,dldt2,dldt3,dldt4,dldt5,dldt6)./prod(blockSize);
        else
            output = cat(3,dldt1,dldt4,dldt5,dldt6)./prod(blockSize);
        end
    else        
        display('Warning: computing numerical gradient');
        output = zeros(size(thetaNn));
        for i=1:size(thetaNn,3) % for every theta coordinate
            for j=1:batchSize
                v = zeros(1,1,size(thetaNn,3),batchSize);
                v(1,1,i,j)=delta;
                thetaNn_plus = thetaNn+v;
                thetaNn_minus = thetaNn-v;

                output(1,1,i,j) = (vl_nnlossaffineNewTheta(thetaNn_plus,thetaGt)-vl_nnlossaffineNewTheta(thetaNn_minus,thetaGt))/(2*delta); % compute numerical derivative using central differences
            end
        end
    end
end


