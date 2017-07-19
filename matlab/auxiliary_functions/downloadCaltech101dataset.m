%% This script downloads the Caltech 101 dataset

mkdir(fullfile(paths.baseDir,'datasets'));
mkdir(fullfile(paths.baseDir,'datasets','caltech-101'));

fprintf('Downloading Caltech-101 dataset ... this may take a while\n') ;
urlwrite('http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz', ...
fullfile(paths.baseDir,'datasets','caltech-101','101_ObjectCategories.tar.gz')) ;
% unzip
fprintf('Unzipping\n') ;
untar(fullfile(paths.baseDir,'datasets','caltech-101','101_ObjectCategories.tar.gz'),fullfile(paths.baseDir,'datasets','caltech-101'));    

fprintf('Downloading Caltech-101 annotations ... this may take a while\n') ;
urlwrite('http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar', ...
fullfile(paths.baseDir,'datasets','caltech-101','Annotations.tar')) ;
% unzip
fprintf('Unzipping\n') ;
untar(fullfile(paths.baseDir,'datasets','caltech-101','Annotations.tar'),fullfile(paths.baseDir,'datasets','caltech-101'));    
% rename some annotation dirs
fprintf('Renaming some annotation directories\n') ;
movefile(fullfile(paths.baseDir,'datasets','caltech-101','Annotations','Airplanes_Side_2'),fullfile(paths.baseDir,'datasets','caltech-101','Annotations','airplanes'))
movefile(fullfile(paths.baseDir,'datasets','caltech-101','Annotations','Faces_2'),fullfile(paths.baseDir,'datasets','caltech-101','Annotations','Faces'))
movefile(fullfile(paths.baseDir,'datasets','caltech-101','Annotations','Faces_3'),fullfile(paths.baseDir,'datasets','caltech-101','Annotations','Faces_easy'))
movefile(fullfile(paths.baseDir,'datasets','caltech-101','Annotations','Motorbikes_16'),fullfile(paths.baseDir,'datasets','caltech-101','Annotations','Motorbikes'))
fprintf('Done renaming\n') ;

% download image pair list
fprintf('Downloading image pair list ... this should be fast\n') ;
urlwrite('http://www.di.ens.fr/willow/research/cnngeometric/other_resources/test_pairs_caltech.mat', ...
fullfile(paths.baseDir,'datasets','caltech-101','test_pairs_caltech.mat')) ;