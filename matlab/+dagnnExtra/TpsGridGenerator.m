classdef TpsGridGenerator < dagnn.Layer
%DAGNN.TPSGRIDGENERATIOR  Generate a thin-plate-spline grid for bilinear resampling
%   This layer maps 1 x 1 x 2*k^2 x N affine transforms to 2 x Ho x Wo x N
%   sampling grids compatible with dagnn.BlilinearSampler, where k^2 is the
%   number of control points, and their X,Y coordinates are stacked along
%   the dim=3 of the input in this way: [X_1,...,X_k^2,Y_1,...,Y_k^2];
%   The coordinates should be normalized into [-1,1]x[-1,1].

 properties
     Ho = 227;
     Wo = 227;
     X=[];
     Y=[];
     Li_w=[];
     Li_a=[];
     square_grid = 1;
     k=3;
     Nkp = []; % number of keypoints (equals k^2 when using square grid)     
     XG=[];
     YG=[];
     useGPU = [];
     lambda = 0;
 end

   methods  ( Access = private )
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
    function outputs = forward(obj, inputs, ~)
        if isempty(obj.useGPU)
            if isa(inputs{1},'gpuArray')
                obj.useGPU=1;
                % move everything to GPU
                obj.Li_w = gpuArray(obj.Li_w);
                obj.Li_a = gpuArray(obj.Li_a);
                obj.XG = gpuArray(obj.XG);
                obj.YG = gpuArray(obj.YG);
                obj.X = gpuArray(obj.X);
                obj.Y = gpuArray(obj.Y);
            end
        end
        
      bSize = size(inputs{1},4);
      % reshape input of control points CP'={X',Y'}
      XpYp = reshape(inputs{1},obj.Nkp,[]);

      Xp = XpYp(:,1:2:end);
      Yp = XpYp(:,2:2:end);
      
      w_x=obj.Li_w*Xp;
      w_y=obj.Li_w*Yp;
      
      a_x=obj.Li_a*Xp;
      a_y=obj.Li_a*Yp;
      
      Xmat = repmat(obj.XG,1,1,1,bSize);
      Ymat = repmat(obj.YG,1,1,1,bSize);
      
      T = @(w,a,X,Y,XG,YG) squeeze(...
          bsxfun(@plus,...
          bsxfun(@times, permute(a(2,:),[1 3 4 2]),XG)+...
          bsxfun(@times, YG, permute(a(3,:),[1 3 4 2])),...
          permute(a(1,:),[1 3 4 2]))+...
          sum(bsxfun(@times,obj.U(sqrt(...
          bsxfun(@minus,XG,permute(X,[3 2 1])).^2+...
          bsxfun(@minus,YG,permute(Y,[3 2 1])).^2)),permute(w,[3 4 1 2])),3));
      
      % compute sampling grid G'={XG',YG'}
        XGp=T(w_x,a_x,obj.X,obj.Y,Xmat,Ymat);
        YGp=T(w_y,a_y,obj.X,obj.Y,Xmat,Ymat);
     
      outputs{1} = permute(cat(4,YGp,XGp),[4 1 2 3]);
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
      bSize = size(inputs{1},4);
      
      XG = repmat(obj.XG,1,1,1,obj.Nkp,bSize); % [Ho x Wo x 1 x bSize]
      YG = repmat(obj.YG,1,1,1,obj.Nkp,bSize); % [Ho x Wo x 1 x bSize]
      
      dLdZ = @(X,Y,XG,YG,dLdZ) ...
      sum(sum(...
      bsxfun(@times,dLdZ,...
        bsxfun(@plus, ...
        bsxfun(@times,ones(size(obj.XG)),permute(obj.Li_a(1,:),[1 3 2 ]))+...
        bsxfun(@times,obj.XG,permute(obj.Li_a(2,:),[1 3 2]))+...
        bsxfun(@times,obj.YG,permute(obj.Li_a(3,:),[1 3 2])),...
                             squeeze(sum(...
                             bsxfun(@times,...
                                obj.U(sqrt(...
                                bsxfun(@minus,XG,permute(X,[3 2 1])).^2+...
                                bsxfun(@minus,YG,permute(Y,[3 2 1])).^2))...
                             ,permute(obj.Li_w,[3 4 1 2])),3))))...
                             ,1),2);
        
      dLdX = dLdZ(obj.X,obj.Y,XG,YG,permute(derOutputs{1}(2,:,:,:),[2 3 1 4]));
      dLdY = dLdZ(obj.X,obj.Y,XG,YG,permute(derOutputs{1}(1,:,:,:),[2 3 1 4]));
      
      derInputs{1}=cat(3,dLdX,dLdY);
      derParams=[];
    end

    % ---------------------------------------------------------------------
    function obj = TpsGridGenerator(varargin)
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
         %   display('using regularization for TPS')
            K = K + obj.lambda*eye(size(K));
        end
        O = zeros(3);

        L = [K P;P' O];
        Li = inv(L);
        
        obj.X=single(X);
        obj.Y=single(Y);
        obj.Li_w=single(Li(1:obj.Nkp,1:obj.Nkp));
        obj.Li_a=single(Li(obj.Nkp+1:end,1:obj.Nkp));
        
        % compute target grid G={XG,YG}
        if isempty(obj.XG) && isempty(obj.YG)
            xi = linspace(-1, 1, obj.Wo);
            yi = linspace(-1, 1, obj.Ho);

            [XG,YG] = meshgrid(xi,yi);
            obj.XG=single(XG);
            obj.YG=single(YG);
        else
            obj.XG = single(obj.XG);
            obj.YG = single(obj.YG);
        end
    end
  end
end
