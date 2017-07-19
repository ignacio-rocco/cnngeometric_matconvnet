function [output] = extra_nnlossdist2(ptsNn,ptsGt,dzdy)

% if length(size(ptsNn))==4
%     batchSize = size(ptsNn,4);
% else 
%     batchSize=1;
% end
if ~iscell(ptsNn)
    if nargin < 3 || isempty(dzdy) % forward pass, output=loss
        % PREVIOUSLY
        % output= sum((ptsNn(:)-ptsGt(:)).^2); %/batchSize;
        output= squeeze(sum((ptsNn-ptsGt).^2,3)); %/batchSize;
    else % backward pass, output=derivative
        output = 2*(ptsNn-ptsGt);
    end
else
    bSize=length(ptsGt);        
    if nargin < 3 || isempty(dzdy) % forward pass, output=loss
        % PREVIOUSLY
        % output= sum((ptsNn(:)-ptsGt(:)).^2); %/batchSize;
        output = zeros(bSize,1,'single');
        if isa(ptsNn{1},'gpuArray')
            output = gpuArray(output);
        end
        for i=1:bSize
            output(i) = sum((ptsNn{i}(:)-ptsGt{i}(:)).^2); %/batchSize;
        end
    else % backward pass, output=derivative
        % allocate memory
        output = ptsNn;        
        for i=1:bSize
            output{i} = 2*(ptsNn{i}-ptsGt{i});
        end
    end    
end

%% Verify numerically

% dldtNum = zeros(6,1);
% loss = sum((ptsNn(:)-ptsGt(:)).^2);
% for i=1:6
%     v=zeros(size(ptsNn)); 
%     v(i)=1e-4;
%     ptsNn2 = double(ptsNn + v);
%     
%     loss2 = sum((ptsNn2(:)-ptsGt(:)).^2);
%     dldtNum(i) = (loss2-loss)/(norm(squeeze(v)));
% end
% dldtNum

