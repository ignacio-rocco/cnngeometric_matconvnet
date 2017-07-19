classdef LossAffine < dagnn.ElementWise
  properties (Transient)
    loss = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = dagnnExtra.extra_nnlossaffine(inputs{1}, inputs{2}, []) ;
      obj.loss = outputs{1}; 
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = dagnnExtra.extra_nnlossaffine(inputs{1}, inputs{2}, derOutputs{1}) ;
      derInputs{2} = [] ;           
      derParams = {} ;
    end

    function reset(obj)
      obj.loss = 0;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = LossAffine(varargin)
      obj.load(varargin) ;
    end
  end
end
