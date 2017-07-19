function net = addConvRegressionModule(net, opts, fsupp_rows, fsupp_cols, featFusedName, thetaName)

gaussianStd = 0.001;

%%% add CONV1 layer (conv+bn+relu)
conv1depth = fsupp_rows*fsupp_cols;

sizeconv1 = [opts.conv1size opts.conv1size conv1depth opts.conv1filters];
convBlock = dagnn.Conv('size', sizeconv1, 'hasBias', true);
net.addLayer('conv1', convBlock, {featFusedName}, {'xconv1'}, {'conv1f', 'conv1b'});
net.params(net.layers(end).paramIndexes(1)).value = randn(sizeconv1,'single')*gaussianStd;
net.params(net.layers(end).paramIndexes(2)).value = randn(sizeconv1(end),1,'single')*gaussianStd;

if isnumeric(opts.batchNormalization) && opts.batchNormalization ==1 || iscell(opts.batchNormalization) && opts.batchNormalization{1}==1
    net.addLayer('bn1', dagnn.BatchNorm(), {net.layers(end).outputs{1}}, {'bn1out'}, {'bn1gamma', 'bn1beta', 'bn1moments'});
    net.params(net.layers(end).paramIndexes(1)).value = single(ones([sizeconv1(end) 1]));
    net.params(net.layers(end).paramIndexes(2)).value = single(zeros([sizeconv1(end) 1]));
    net.params(net.layers(end).paramIndexes(3)).value = single(zeros([sizeconv1(end) 2]));
end

net.addLayer('relu1', dagnn.ReLU(), {net.layers(end).outputs{1}}, {'relu1out'}, {});

%%% add CONV2 layer (conv+bn+relu)
sizeconv2 = [opts.conv2size opts.conv2size opts.conv1filters opts.conv2filters];
convBlock2 = dagnn.Conv('size', sizeconv2, 'hasBias', true) ;
net.addLayer('conv2', convBlock2, {net.layers(end).outputs{1}}, {'conv2out'}, {'conv2f', 'conv2b'});
net.params(net.layers(end).paramIndexes(1)).value = randn(sizeconv2,'single')*gaussianStd;
net.params(net.layers(end).paramIndexes(2)).value = randn(sizeconv2(end),1,'single')*gaussianStd;

if isnumeric(opts.batchNormalization) && opts.batchNormalization ==1 || iscell(opts.batchNormalization) && opts.batchNormalization{2}==1
    net.addLayer('bn2', dagnn.BatchNorm(), {net.layers(end).outputs{1}}, {'bn2out'}, {'bn2gamma', 'bn2beta', 'bn2moments'});
    net.params(net.layers(end).paramIndexes(1)).value = single(ones([sizeconv2(end) 1]));
    net.params(net.layers(end).paramIndexes(2)).value = single(zeros([sizeconv2(end) 1]));
    net.params(net.layers(end).paramIndexes(3)).value = single(zeros([sizeconv2(end) 2]));
end

% add relu
net.addLayer('relu2', dagnn.ReLU(), {net.layers(end).outputs{1}}, {'relu2out'}, {});
    
%%% add CONV3 layer (actually a fully connected layer)
sizeconv3 = [opts.conv3size opts.conv3size opts.conv2filters opts.conv3filters];
convBlock3 = dagnn.Conv('size', sizeconv3, 'hasBias', true) ;      
net.addLayer('conv3', convBlock3, {net.layers(end).outputs{1}}, {thetaName}, {'conv3f', 'conv3b'});
net.params(net.layers(end).paramIndexes(1)).value = randn(sizeconv3,'single')*gaussianStd;
net.params(net.layers(end).paramIndexes(2)).value = randn(sizeconv3(end),1,'single')*gaussianStd;



