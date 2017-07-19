function anetStruct = netNamePrefix(anetStruct,layerPrefix, varPrefix, paramPrefix, shiftParam)

if nargin<5
    shiftParam=1;
end
if nargin<4
    paramPrefix = layerPrefix;
end
if nargin<3
    varPrefix = layerPrefix;
end

dag = dagnn.DagNN.loadobj(anetStruct);

% rename layer
for i=1:length(anetStruct.layers)
   % layer name
   anetStruct.layers(i).name = [layerPrefix anetStruct.layers(i).name];
   % layer inputs
   for j=1:length(anetStruct.layers(i).inputs)
       anetStruct.layers(i).inputs{j}=[varPrefix anetStruct.layers(i).inputs{j}];
   end
   % layer outputs
   for j=1:length(anetStruct.layers(i).outputs)
       anetStruct.layers(i).outputs{j}=[varPrefix anetStruct.layers(i).outputs{j}];
   end
   % layer params
   if ~isempty(anetStruct.layers(i).params)
       for j=1:length(anetStruct.layers(i).params)
           if dag.getParamIndex(anetStruct.layers(i).params{j})>=shiftParam
            anetStruct.layers(i).params{j}=[paramPrefix anetStruct.layers(i).params{j}];
           end
       end
   end
end

% rename vars
for i=1:length(anetStruct.vars)
    anetStruct.vars(i).name = [varPrefix anetStruct.vars(i).name];
end

% rename params
for i=shiftParam:length(anetStruct.params)
    anetStruct.params(i).name = [paramPrefix anetStruct.params(i).name];
end