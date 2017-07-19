% =========================================================================
%
% Author: Ignacio Rocco 
%
% This script demonstrates the training procedure described in cnngeometric
% Refer to README.md for setup instructions, and to our project page for
% additional information: http://www.di.ens.fr/willow/research/cnngeometric/
%
% =========================================================================

%% ======================== Setup environment and download training dataset

% setup paths
setup;

% define path to training dataset
paths.trValdatasetPath = '/sequoia/data1/iroccosp/datasets/instance/pascal-berkeley-voc11';

% download the Pascal-VOC 2011 for training if not there
if isempty(dir(paths.trValdatasetPath))
    downloadPascal2011dataset ;
end

% download pretrained VGG-16 model
if isempty(dir(fullfile(paths.baseDir,'training','imagenet-vgg-verydeep-16.mat')))
    downloadPretrainedVGG16 ;
end

%% ================================================== Load training options 

% load CNN training options (topts)
load(fullfile(paths.baseDir,'training','training_options','aff_pascal','topts.mat'));

% use GPU?
topts.gpus=1;

% train
trainNetwork(paths,topts);

