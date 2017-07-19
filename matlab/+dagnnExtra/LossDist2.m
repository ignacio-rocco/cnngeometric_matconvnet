classdef LossDist2 < dagnn.ElementWise
  properties (Transient)
    loss = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = dagnnExtra.extra_nnlossdist2(inputs{1}, inputs{2}, []) ;
      obj.loss = outputs{1}; 
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = dagnnExtra.extra_nnlossdist2(inputs{1}, inputs{2}, derOutputs{1}) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
        obj.loss = 0;
    end

    function obj = LossDist2(varargin)
      obj.load(varargin) ;
    end
  end
end
