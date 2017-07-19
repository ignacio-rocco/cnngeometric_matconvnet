%% This script downloads the Proposal Flow dataset and test pairs

mkdir(fullfile(paths.baseDir,'datasets'));

fprintf('Downloading PF-dataset ... this may take a bit\n') ;
urlwrite('http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip', ...
fullfile(paths.baseDir,'datasets','PF-dataset.zip')) ;
% unzip
fprintf('Unzipping\n') ;
unzip(fullfile(paths.baseDir,'datasets','PF-dataset.zip'),fullfile(paths.baseDir,'datasets'));    
% download image pair list
fprintf('Downloading image pair list ... this should be fast\n') ;
urlwrite('http://www.di.ens.fr/willow/research/cnngeometric/other_resources/test_pairs_pf.mat', ...
fullfile(paths.baseDir,'datasets','PF-dataset','test_pairs_pf.mat')) ;