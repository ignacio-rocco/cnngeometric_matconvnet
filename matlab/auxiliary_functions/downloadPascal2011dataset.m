%% This script downloads the Pascal VOC 2011 dataset

mkdir(fullfile(paths.baseDir,'datasets'));
mkdir(fullfile(paths.baseDir,'datasets','pascal-voc11'));

fprintf('Downloading Pascal VOC 2011 dataset (~1.6G) ... this may take a while\n') ;
urlwrite('http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar', ...
fullfile(paths.baseDir,'datasets','pascal-voc11','VOCtrainval_25-May-2011.tar')) ;
% unzip
fprintf('Unzipping\n') ;
untar(fullfile(paths.baseDir,'datasets','pascal-voc11','VOCtrainval_25-May-2011.tar'),fullfile(paths.baseDir,'datasets','pascal-voc11'));    
