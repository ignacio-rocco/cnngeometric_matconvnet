function im1_batch2 = subMeanAndReshape(im1_batch,subMean)

if nargin<2
    subMean=1;
end

avim = cat(3,122.7395,114.8967,101.5919);
[rows,cols,channels] = size(im1_batch{1}(:,:,:));
N = length(im1_batch);

im1_batch2 = zeros(rows,cols,3,N,'single');

% substract mean and copy
for i=1:N
    if size(im1_batch{i},3)==1
        im1_batch{i} = repmat(im1_batch{i},1,1,3);
    end
    if subMean==1
        try
            im1_batch2(:,:,:,i) = bsxfun(@minus,im1_batch{i},avim);
        catch
            size(im1_batch{i})
            im1_batch2(:,:,:,i) = bsxfun(@minus,im1_batch{i},avim);
        end
    else
        im1_batch2(:,:,:,i) = im1_batch{i};
    end
end

% % substract mean
% for i=1:N
%     im1_batch{i} = bsxfun(@minus,im1_batch{i},avim);
% end
% 
% % % convert from cell to matrix
% im1_batch=permute(reshape(permute(cell2mat(im1_batch),[2 3 1]),cols,3,rows,N),[3 1 2 4]);
