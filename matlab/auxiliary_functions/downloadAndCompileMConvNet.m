%% This script downloads MatConvNet beta 20

mkdir(fullfile(paths.baseDir,'datasets'));
mkdir(fullfile(paths.baseDir,'datasets','caltech-101'));

fprintf('Downloading MatConvNet beta 20...\n') ;
urlwrite('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta20.tar.gz', ...
fullfile(paths.baseDir,'matlab','matconvnet-1.0-beta20.tar.gz')) ;
% unzip
fprintf('Unzipping\n') ;
untar(fullfile(paths.baseDir,'matlab','matconvnet-1.0-beta20.tar.gz'),fullfile(paths.baseDir,'matlab'));    

% ask for confirmation before compiling
answer = input('You are about to compile MatConvNet with default options (e.g. CPU, no GPU acceleration). Would you like to continue?, Y/N [Y]:','s');
    if strcmp(answer,'y')==1 || strcmp(answer,'Y')==1 || isempty(answer)
    % if you get a warning about your gcc version being not compatible,
    % install the suggested gcc-4.9 and replace flags.cc = {} with
    % flags.cc = {'GCC=''/usr/bin/gcc-4.9'''} in vl_compilenn (line 306 for
    % MatConvNet beta 20)
    warning('If you get a warning about your gcc version being not compatible, install the suggested gcc-4.9 and replace flags.cc = {} with {''GCC=''''/usr/bin/gcc-4.X''''''} in vl_compilenn and recompile (line 306 for MatConvNet beta 20)');
    run matlab/matconvnet-1.0-beta20/matlab/vl_compilenn
end