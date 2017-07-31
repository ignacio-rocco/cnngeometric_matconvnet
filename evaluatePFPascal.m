% =========================================================================
%
% Author: Ignacio Rocco 
%
% This script allows to evaluate the trained models on the different datasets
%
% =========================================================================

%% ==================================== Setup environment and select models

setup;
% download the pre-trained models from the web if needed
downloadPretrainedModels(paths,'aff') ; % download affine CNN
downloadPretrainedModels(paths,'aff_larger_range') ; % download affine CNN
downloadPretrainedModels(paths,'tps') ; % download TPS CNN

evalopts=struct();
evalopts.affnet = 'aff';
%evalopts.affnet2 = [];  % you can use 'aff_larger_range' to try the affine ensemble
evalopts.affnet2 = 'aff_larger_range';  % you can use 'aff_larger_range' to try the affine ensemble
evalopts.tpsnet = 'tps';

evalopts.netmodelpath = paths.trainedModels; % define CNN model base path

% default paths to datasets
paths.pfPascalPath = fullfile(paths.baseDir,'datasets','PF-dataset-PASCAL'); % PF dataset

%% ====================================== Evaluate on Proposal Flow dataset

% download Proposal Flow dataset if needed
if ~exist(fullfile(paths.pfPascalPath), 'file')
    downloadPFPascaldataset;
end
% evaluate on Proposal Flow dataset
resultsPF = evaluatePropFlowPascal(paths,evalopts);

display(['Mean PCK on Proposal Flow: ', num2str(resultsPF.meanPck)]);

%% ======================================== Evaluate on Caltech-101 dataset

% download Caltech dataset if needed
if ~exist(fullfile(paths.caltech101Path), 'file')
    downloadCaltech101dataset;
end

% evaluate on Proposal Flow dataset
resultsCaltech = evaluateCaltech(paths,evalopts);

display(['Mean IoU on Caltech: ', ... 
    num2str(mean(resultsCaltech.IoUafftps(find(resultsCaltech.validPairs))))]);
display(['Mean LT-ACC on Caltech: ', ...
    num2str(mean(resultsCaltech.LTACCafftps(find(resultsCaltech.validPairs))))]);
display(['Mean LOC-ERR on Caltech: ', ...
    num2str(mean(resultsCaltech.LOCERRafftps(find(resultsCaltech.validPairs))))]);