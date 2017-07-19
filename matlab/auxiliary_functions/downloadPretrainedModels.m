%% Download pretrained models
% model = 'aff'/'tps'/'aff_larger_range'
function downloadPretrainedModels(paths,model)

if strcmp(model,'aff') && ~exist(fullfile(paths.trainedModels,'aff','net.mat'), 'file')
    mkdir(paths.trainedModels);
    mkdir(fullfile(paths.trainedModels,'aff'));
	fprintf('Downloading affine model ... this may take a bit\n') ;
	urlwrite('http://www.di.ens.fr/willow/research/cnngeometric/trained_models/aff/net.mat', ...
    fullfile(paths.trainedModels,'aff','net.mat')) ;
    fprintf('Done\n');
end

if strcmp(model,'aff_larger_range') && ~exist(fullfile(paths.trainedModels,'aff_larger_range','net.mat'), 'file')
    mkdir(paths.trainedModels);
    mkdir(fullfile(paths.trainedModels,'aff_larger_range'));
	fprintf('Downloading aff_larger_range model ... this may take a bit\n') ;
	urlwrite('http://www.di.ens.fr/willow/research/cnngeometric/trained_models/aff_larger_range/net.mat', ...
    fullfile(paths.trainedModels,'aff_larger_range','net.mat')) ;
    fprintf('Done\n');
end

if strcmp(model,'tps') && ~exist(fullfile(paths.trainedModels,'tps','net.mat'), 'file')
    mkdir(paths.trainedModels);
    mkdir(fullfile(paths.trainedModels,'tps'));
	fprintf('Downloading TPS model ... this may take a bit\n') ;
	urlwrite('http://www.di.ens.fr/willow/research/cnngeometric/trained_models/tps/net.mat', ...
	fullfile(paths.trainedModels,'tps','net.mat')) ;
    fprintf('Done\n');
end