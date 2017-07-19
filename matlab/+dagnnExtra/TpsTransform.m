classdef TpsTransform < dagnn.Layer
%DAGNN.TPSTRANSFORM  Transform given points using TPS model of given params
%   input{1} is matrix of TPS parameters: k^2 is the
%   number of control points, and their X,Y coordinates are stacked along
%   the dim=3 of the input in this way: [X_1,...,X_k^2,Y_1,...,Y_k^2];
%   The coordinates should be normalized into [-1,1]x[-1,1].
%   input{2} is cell array of matrices containing [X Y] coords of points


 properties
     Ho = 227;
     Wo = 227;
     X=[];
     Y=[];
     Li_w=[];
     Li_a=[];
     square_grid = 1;
     k=10;
     Nkp = []; % number of keypoints (equals k^2 when using square grid)     
     useGPU = [];
     T=[];
     dLdZ=[];
     thetaZeroCentered=0;
     lambda = 0;
 end

   methods  %( Access = private )
%         function u = U(obj,r)
%             u = r.^2.*log(r.^2);
%             u(r==0)=0; % fix 0*log(0) nan
%         end
        function u = U(obj,r)
            r(r==0)=1; % fix 0*log(0) nan
            u = r.^2.*log(r.^2);
            %u(r==0)=0; % fix 0*log(0) nan
        end
    end

  methods
    function outputs = forward(obj, inputs, ~) % inputs{1}, TPS grid params, inputs{2} cell of points to transform
        if isempty(obj.useGPU)
            if isa(inputs{1},'gpuArray')
                obj.useGPU=1;
                % move everything to GPU
                obj.Li_w = gpuArray(obj.Li_w);
                obj.Li_a = gpuArray(obj.Li_a);
                obj.X = gpuArray(obj.X);
                obj.Y = gpuArray(obj.Y);
            end
        end
        
      bSize = size(inputs{1},4);
      % reshape input of control points CP'={X',Y'}
      XpYp = reshape(inputs{1},obj.Nkp,[]);

      Xp = XpYp(:,1:2:end);
      Yp = XpYp(:,2:2:end);
      
      % shift values
      if obj.thetaZeroCentered
          Xp = Xp+repmat(obj.X,[1 bSize]);
          Yp = Yp+repmat(obj.Y,[1 bSize]);
      end
      
      w_x=obj.Li_w*Xp;
      w_y=obj.Li_w*Yp;
      
      a_x=obj.Li_a*Xp;
      a_y=obj.Li_a*Yp;
      
      % allocate memory
      outputs{1}=inputs{2};
      
      for i=1:bSize
          % compute x component
          outputs{1}{i}(:,1) = obj.T(w_x(:,i),a_x(:,i),obj.X,obj.Y,inputs{2}{i}(:,1),inputs{2}{i}(:,2)); 
          % compute y component
          outputs{1}{i}(:,2) = obj.T(w_y(:,i),a_y(:,i),obj.X,obj.Y,inputs{2}{i}(:,1),inputs{2}{i}(:,2)); 
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
      
      bSize = size(inputs{1},4);
      
      %XG = repmat(obj.XG,1,1,1,obj.Nkp,bSize); % [Ho x Wo x 1 x bSize]
      %YG = repmat(obj.YG,1,1,1,obj.Nkp,bSize); % [Ho x Wo x 1 x bSize]

      % allocate memory
      derInputs{1} = inputs{1};
      
      for i=1:bSize
          dLdX = obj.dLdZ(obj.X,obj.Y,inputs{2}{i}(:,1), inputs{2}{i}(:,2), derOutputs{1}{i}(:,1));
          dLdY = obj.dLdZ(obj.X,obj.Y,inputs{2}{i}(:,1), inputs{2}{i}(:,2), derOutputs{1}{i}(:,2));      
          
       % dLdX = dLdZ(obj.X,obj.Y,inputs{2}{i}(:,1), inputs{2}{i}(:,2), obj.Li_w, obj.Li_a, derOutputs{1}{i}(:,1));
       % dLdY = dLdZ(obj.X,obj.Y,inputs{2}{i}(:,1), inputs{2}{i}(:,2), obj.Li_w, obj.Li_a, derOutputs{1}{i}(:,2));      
       % dLdX = permute(dLdX,[2 3 1]);
       % dLdY = permute(dLdY,[2 3 1]);
          derInputs{1}(:,:,:,i)=cat(3,dLdX,dLdY);
      end
      derInputs{2}=[]; % we don't care about the derivative with respect to the ground truth points
      derParams=[];
    end

    % ---------------------------------------------------------------------
    function obj = TpsTransform(varargin)
      obj.load(varargin) ;
      
        % generate grid of control points CP={X,Y}
        if obj.square_grid==1
            [X,Y]=meshgrid(linspace(-1, 1, obj.k),linspace(-1, 1, obj.k));
            X=X(:);
            Y=Y(:);
            obj.Nkp = obj.k^2;
        else % expect keypoints to be given by user 
            X = obj.X;
            Y = obj.Y;
            obj.Nkp = length(X);
        end
        P=[ones(size(X)) X Y];

        PXmat = repmat(X,1,length(X));
        PYmat = repmat(Y,1,length(Y));

        K = obj.U(sqrt((PXmat-PXmat').^2+(PYmat-PYmat').^2));
        if obj.lambda~=0 % apply regularization
            display('using regularization for TPS')
            K = K + obj.lambda*eye(size(K));
        end
        
        O = zeros(3);

        L = [K P;P' O];
        Li = inv(L);
        
        obj.X=single(X);
        obj.Y=single(Y);
        obj.Li_w=single(Li(1:obj.Nkp,1:obj.Nkp));
        obj.Li_a=single(Li(obj.Nkp+1:end,1:obj.Nkp));     
        
        obj.T = @(w,a,X,Y,XG,YG) squeeze(...
          bsxfun(@plus,...
          bsxfun(@times, permute(a(2,:),[1 3 4 2]),XG)+...
          bsxfun(@times, YG, permute(a(3,:),[1 3 4 2])),...
          permute(a(1,:),[1 3 4 2]))+...
          sum(bsxfun(@times,obj.U(sqrt(...
          bsxfun(@minus,XG,permute(X,[3 2 1])).^2+...
          bsxfun(@minus,YG,permute(Y,[3 2 1])).^2)),permute(w,[3 4 1 2])),3));
      
        obj.dLdZ = @(X,Y,XG,YG,dLdZ) ...
        sum(sum(...
        bsxfun(@times,dLdZ,...
        bsxfun(@plus, ...
        bsxfun(@times,ones(size(XG)),permute(obj.Li_a(1,:),[1 3 2 ]))+...
        bsxfun(@times,XG,permute(obj.Li_a(2,:),[1 3 2]))+...
        bsxfun(@times,YG,permute(obj.Li_a(3,:),[1 3 2])),...
                             sum(...
                             bsxfun(@times,...
                                obj.U(sqrt(...
                                bsxfun(@minus,XG,permute(X,[2 1])).^2+...
                                bsxfun(@minus,YG,permute(Y,[2 1])).^2))...
                             ,permute(obj.Li_w,[3 1 2])),2)))...
                             ,1),2);
% 
%       obj.dLdZ = @(X,Y,XG,YG,dLdZ) ...
%       sum(sum(...
%       bsxfun(@times,dLdZ,...
%         bsxfun(@plus, ...
%         bsxfun(@times,ones(size(XG)),permute(obj.Li_a(1,:),[1 3 2 ]))+...
%         bsxfun(@times,XG,permute(obj.Li_a(2,:),[1 3 2]))+...
%         bsxfun(@times,YG,permute(obj.Li_a(3,:),[1 3 2])),...
%                              permute(sum(...
%                              bsxfun(@times,...
%                                 obj.U(sqrt(...
%                                 bsxfun(@minus,XG,permute(X,[3 2 1])).^2+...
%                                 bsxfun(@minus,YG,permute(Y,[3 2 1])).^2))...
%                              ,permute(obj.Li_w,[3 4 1 2])),3),[1 2 4 3])))...
%                              ,1),2);
%         

    end
  end
end
