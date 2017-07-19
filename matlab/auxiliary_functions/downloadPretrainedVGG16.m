%% This script downloads a pretrained VGG-16 model

mkdir(fullfile(paths.baseDir,'training'));

fprintf('Downloading a pretrained VGG-16 model (~500M)... this may take a while\n') ;
urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', ...
fullfile(paths.baseDir,'training','imagenet-vgg-verydeep-16.mat')) ;