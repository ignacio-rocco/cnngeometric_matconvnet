classdef Correlate < dagnn.Filter
  properties
    %size = [0 0 0 0]
    %hasBias = true
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(obj, inputs, params)
      [hA,wA,d,batchsize]= size(inputs{1});
      [hB,wB,d,batchsize]= size(inputs{2});
      inputs1filter = reshape(permute(inputs{1},[3,1,2,4]),1,1,d,hA*wA,[]);
      if isa(inputs{1},'single')
        outputs{1} = zeros(hB,wB,hA*wA,batchsize,'single');
      else
        outputs{1} = gpuArray(zeros(hB,wB,hA*wA,batchsize,'single'));
      end
      for i=1:batchsize
        outputs{1}(:,:,:,i) = vl_nnconv(inputs{2}(:,:,:,i), inputs1filter(:,:,:,:,i),[]) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [h,w,d,batchsize]= size(inputs{1});
      inputs1filter = reshape(permute(inputs{1},[3,1,2,4]),1,1,d,h*w,[]);
      if isa(inputs{1},'single')
        derInputs{1} = zeros(h,w,d,batchsize,'single');
        derInputs{2} = zeros(h,w,d,batchsize,'single');
      else
        derInputs{1} = gpuArray(zeros(h,w,d,batchsize,'single'));
        derInputs{2} = gpuArray(zeros(h,w,d,batchsize,'single'));
      end
      for i=1:batchsize
      [dI2, dI1, ~] = vl_nnconv(...
        inputs{2}(:,:,:,i), inputs1filter(:,:,:,:,i), [], derOutputs{1}(:,:,:,i)) ;
        derInputs{1}(:,:,:,i) = permute(reshape(dI1,size(inputs{1},3),size(inputs{1},1),size(inputs{1},2),[]),[2 3 1 4]);
        derInputs{2}(:,:,:,i) = dI2;
      end
      derParams = {} ;
    end

%     function kernelSize = getKernelSize(obj)
%       kernelSize = obj.size(1:2) ;
%     end

%     function outputSizes = getOutputSizes(obj, inputSizes)
%       outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
%       outputSizes{1}(3) = obj.size(4) ;
%     end

%     function params = initParams(obj)
%       sc = sqrt(2 / prod(obj.size(1:3))) ;
%       params{1} = randn(obj.size,'single') * sc ;
%       if obj.hasBias
%         params{2} = zeros(obj.size(4),1,'single') * sc ;
%       end
%     end

%     function set.size(obj, ksize)
%       % make sure that ksize has 4 dimensions
%       ksize = [ksize(:)' 1 1 1 1] ;
%       obj.size = ksize(1:4) ;
%     end

    function obj = Correlate(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      %obj.size = obj.size ;
      %obj.stride = obj.stride ;
      %obj.pad = obj.pad ;
    end
  end
end
