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

%% ========================================== Choose dataset to evaluate on

datasets = {'caltech-101','PF-dataset','PF-dataset-PASCAL','TSS_CVPR2016'};

evaluationDataset = datasets{4}; 

%% ======================================== Evaluate on Caltech-101 dataset
if strcmp(evaluationDataset,'caltech-101')==1

    % default path to dataset
    paths.caltech101Path = fullfile(paths.baseDir,'datasets','caltech-101'); % Caltech-101 dataset

    % download Caltech dataset if needed
    if ~exist(fullfile(paths.caltech101Path), 'file')
        downloadCaltech101dataset;
    end

    % evaluate on Proposal Flow dataset
    resultsCaltech = evaluateCaltech(paths,evalopts);

    % if ~isempty(evalopts.tpsnet)
        display(['Mean IoU on Caltech: ', ... 
            num2str(mean(resultsCaltech.IoUafftps(find(resultsCaltech.validPairs))))]);
        display(['Mean LT-ACC on Caltech: ', ...
            num2str(mean(resultsCaltech.LTACCafftps(find(resultsCaltech.validPairs))))]);
        display(['Mean LOC-ERR on Caltech: ', ...
            num2str(mean(resultsCaltech.LOCERRafftps(find(resultsCaltech.validPairs))))]);
      else
        display(['Mean IoU on Caltech: ', ... 
            num2str(mean(resultsCaltech.IoUaff(find(resultsCaltech.validPairs))))]);
        display(['Mean LT-ACC on Caltech: ', ...
            num2str(mean(resultsCaltech.LTACCaff(find(resultsCaltech.validPairs))))]);
        display(['Mean LOC-ERR on Caltech: ', ...
            num2str(mean(resultsCaltech.LOCERRaff(find(resultsCaltech.validPairs))))]);
       end

end
%% ============================== Evaluate on Proposal Flow (WILLOW) dataset
if strcmp(evaluationDataset,'PF-dataset')==1

    % default path to dataset
    paths.pfPath = fullfile(paths.baseDir,'datasets','PF-dataset'); % PF dataset

    % download Proposal Flow dataset if needed
    if ~exist(fullfile(paths.pfPath), 'file')
        downloadPFdataset;
    end
    % evaluate on Proposal Flow dataset
    resultsPF = evaluatePropFlow(paths,evalopts);

    display(['Mean PCK on Proposal Flow: ', num2str(resultsPF.meanPck)]);

end
%% ============================= Evaluate on Proposal Flow (PASCAL) dataset
if strcmp(evaluationDataset,'PF-dataset-PASCAL')==1

    % default path to dataset
    paths.pfPascalPath = fullfile(paths.baseDir,'datasets','PF-dataset-PASCAL'); % PF dataset

    % download Proposal Flow PASCAL dataset if needed
    if ~exist(fullfile(paths.pfPascalPath), 'file')
        downloadPFPascaldataset;
    end
    % evaluate on Proposal Flow PASCAL dataset
    resultsPF_PASCAL = evaluatePropFlowPascal(paths,evalopts);

    display(['Mean PCK on Proposal Flow PASCAL: ', num2str(resultsPF_PASCAL.meanPck)]);

end
%% ====================================== Evaluate on TSS dataset
if strcmp(evaluationDataset,'TSS_CVPR2016')==1

    % default path to dataset
    paths.TSSPath = fullfile(paths.baseDir,'datasets','TSS_CVPR2016'); % PF dataset

    % download TSS dataset if needed
    if ~exist(fullfile(paths.TSSPath), 'file')
        downloadTSSdataset;
    end
    % evaluate on TSS dataset
    resultsTSS = evaluateTSS(paths,evalopts);
    
    display(['Mean PCK on TSS-FG3Dcar: ', num2str(resultsTSS.meanPckFG3DCar)]);
    display(['Mean PCK on TSS-PASCAL: ', num2str(resultsTSS.meanPckPASCAL)]);
    display(['Mean PCK on TSS-JODS: ', num2str(resultsTSS.meanPckJODS)]);
    display(['Mean PCK on TSS-All: ', num2str(resultsTSS.meanPckAll)]);
    
end
