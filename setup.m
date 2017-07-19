% =========================================================================
%
% Author: Ignacio Rocco 
% 
% This script defines specifies data folders and result folders to be used
% at evaluation and training
%
% =========================================================================

%% ========================================================= Base directory
paths.baseDir = pwd; % path to cnn_geometric main folder

%% ============================================================= Evaluation

% trained models
paths.trainedModels = fullfile(paths.baseDir,'trained_models');
% default paths to datasets
paths.pfPath = fullfile(paths.baseDir,'datasets','PF-dataset'); % PF dataset
paths.caltech101Path = fullfile(paths.baseDir,'datasets','caltech-101'); % Caltech-101 dataset

%% =============================================================== Training

paths.results = fullfile(paths.baseDir,'results');

%% ===================================== Add paths to scripts and functions

addpath(fullfile(paths.baseDir,'matlab'))
addpath(fullfile(paths.baseDir,'matlab','auxiliary_functions'))
addpath(fullfile(paths.baseDir,'trained_models'))
addpath(fullfile(paths.baseDir,'evaluation','PF_willow'))
addpath(fullfile(paths.baseDir,'evaluation','caltech-101'))
addpath(fullfile(paths.baseDir,'training'))
addpath(fullfile(paths.baseDir,'training','training_code'))

% get name of the MatConvNet version uncompressed to
% ./matlab/matconvnet-.../
% eg. ./matconvnet/matconvnet-1.0-beta20
matconvnetfolder = dir(fullfile(paths.baseDir,'matlab','matconvnet-*'));

%% ======================================== download and compile MatConvNet
if isempty(matconvnetfolder)
    answer = input('MatConvNet folder was not found. Would you like to download and compile it?, Y/N [Y]:','s');
    if strcmp(answer,'y')==1 || strcmp(answer,'Y')==1 || isempty(answer)
        downloadAndCompileMConvNet ;
    end
    matconvnetfolder = dir(fullfile(paths.baseDir,'matlab','matconvnet-*'));
end

addpath(fullfile(paths.baseDir,'matlab',matconvnetfolder(1).name,'matlab'));
run vl_setupnn

