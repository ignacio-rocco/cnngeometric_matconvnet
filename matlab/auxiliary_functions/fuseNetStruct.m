function anetStructFused = fuseNetStruct(anetStruct1, anetStruct2)

if isempty(anetStruct1)
    anetStructFused = anetStruct2;
    return;
end

anetStructFused=anetStruct1;

% fuse layers
for i=1:length(anetStruct2.layers)
    % check if layer exists already
    if isempty(find(strcmp({anetStructFused.layers.name}, anetStruct2.layers(i).name)))
        % if not, add
        anetStructFused.layers(end+1)=anetStruct2.layers(i);        
    end
end

% fuse vars
for i=1:length(anetStruct2.vars)
    % check if var exists already
    if isempty(find(strcmp({anetStructFused.vars.name}, anetStruct2.vars(i).name)))
        % if not, add
        anetStructFused.vars(end+1)=anetStruct2.vars(i);        
    end
end

% fuse params
for i=1:length(anetStruct2.params)
    % check if var params already
    if isempty(find(strcmp({anetStructFused.params.name}, anetStruct2.params(i).name)))
        % if not, add
        anetStructFused.params(end+1)=anetStruct2.params(i);        
    end
end