% =========================================================================
%
% Author: Ignacio Rocco 
%
% This script demonstrates cnngeometric on the Proposal Flow dataset
% Refer to README.md for setup instructions, and to our project page for
% additional information: http://www.di.ens.fr/willow/research/cnngeometric/
%
% =========================================================================

%% =========================== Setup environment and load networks and data

% setup paths
setup;

% download the pre-trained models from the web
downloadPretrainedModels(paths,'aff') ; % download affine CNN
downloadPretrainedModels(paths,'tps') ; % download TPS CNN
% load pre-trained models
netAff = load(fullfile(paths.trainedModels,'aff','net.mat')) ;
netAff = netAff.net ;
netTps = load(fullfile(paths.trainedModels,'tps','net.mat')) ;
netTps = netTps.net ;
% remove loss layer, which is not needed for evaluation
netAff.removeLayer(netAff.layers(end).name) ;
netTps.removeLayer(netTps.layers(end).name) ;
% store theta output in memory
netAff.vars(netAff.getVarIndex('theta')).precious=1 ;
netTps.vars(netTps.getVarIndex('theta')).precious=1 ;

% Proposal Flow dataset base path
pfPath = paths.pfPath ;

% download Proposal Flow dataset
if ~exist(fullfile(pfPath), 'file')
    downloadPFdataset;
end

% load pair list
load(fullfile(pfPath,'test_pairs_pf.mat')) ;
N = length(pairs.cat) ; % num. of images

%% ========================================= Load and process an image pair

% obtain and preprocess an image pair
idx = randi(N) ; % random pair index
imA = imread(fullfile(pfPath,pairs.imageA{idx})) ;
[hA, wA] = size(imA(:,:,1));
imA_ = preprocessImage(imA) ; 
imB = imread(fullfile(pfPath,pairs.imageB{idx})) ;
[hB, wB] = size(imB(:,:,1));
imB_ = preprocessImage(imB) ; 

% display
figure(1);clf;imshow([imresize(imA,size(imB(:,:,1))) imB]); title('source image')

% run the affine CNN
netAff.eval({'AN1input',imA_,'AN2input',imB_}) ;
thetaAff = netAff.vars(netAff.getVarIndex('theta')).value ;

% warp image with affine transformation
agg = dagnn.AffineGridGenerator('Ho',hB,'Wo',wB) ;
bs = dagnn.BilinearSampler ; 
grAff = agg.forward({thetaAff}) ;
imAaff = bs.forward({single(imA),grAff{1}}) ; 
imAaff = imAaff{1} ;
imAaff_ = preprocessImage(imAaff) ;

% run the TPS CNN
netTps.eval({'AN1input',imAaff_,'AN2input',imB_}) ;
thetaTps = netTps.vars(netTps.getVarIndex('theta')).value ;

% warp image with TPS transformation
k = sqrt(length(thetaTps)/2) ; % TPS grid size k x k
lambda = 0; % TPS regularization
tgg = dagnnExtra.TpsGridGenerator('Ho',hB,'Wo',wB,'k',k,'lambda',lambda) ;
bs = dagnn.BilinearSampler ;  
grTps = tgg.forward({thetaTps}) ;

% compose TPS and aff transformations
grAffTps = composeAffTps(grAff,grTps);

% compute result image
imAafftps = bs.forward({single(imA),grAffTps}) ; 
imAafftps = imAafftps{1};

% display results
figure(2);clf;imshow([uint8(imAaff) uint8(imAafftps)]); title('alignment with affine/affine+tps model')


