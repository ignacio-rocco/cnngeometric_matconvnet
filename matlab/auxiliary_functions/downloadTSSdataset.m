%% This script downloads the Proposal Flow dataset and test pairs

mkdir(fullfile(paths.baseDir,'datasets'));

fprintf('Downloading TSS dataset ... this may take a bit\n');
urlwrite('http://www.hci.iis.u-tokyo.ac.jp/datasets/data/JointCorrCoseg/TSS_CVPR2016.zip', ...
fullfile(paths.baseDir,'datasets','TSS_CVPR2016.zip'));
% unzip
fprintf('Unzipping\n');
unzip(fullfile(paths.baseDir,'datasets','TSS_CVPR2016.zip'),fullfile(paths.baseDir,'datasets'));    
%
fprintf('Downloading TSS EvaluationKit ... this may take a bit\n');
urlwrite('https://github.com/t-taniai/TSS_CVPR2016_EvaluationKit/archive/master.zip', ...
fullfile(paths.baseDir,'evaluation','TSS','TSS_CVPR2016_EvaluationKit.zip'));
% unzip
fprintf('Unzipping\n');
unzip(fullfile(paths.baseDir,'evaluation','TSS','TSS_CVPR2016_EvaluationKit.zip'),fullfile(paths.baseDir,'evaluation','TSS'));    
