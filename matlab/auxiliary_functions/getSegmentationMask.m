function [mask,hf] = getSegmentationMask(imA, annA,hf)

if isempty(hf) || hf.isvalid==0
    hf = figure('color','white','units','normalized','position',[.1 .1 .8 .8]); 
    figure(hf);
else
    set(0, 'CurrentFigure', hf);
end
clf;image(zeros(size(imA))); 
set(gca,'units','pixels','position',[5 5 size(imA,2)-1 size(imA,1)-1],'visible','off')
hold on
% draw polygon
fill(annA.obj_contour(1,:)+annA.box_coord(3),annA.obj_contour(2,:)+annA.box_coord(1),'w')
% Capture the image 
% Note that the size will have changed by about 1 pixel 
tim = getframe(gca); 
% Extract the cdata
%mask = tim.cdata(1:size(imA,1),1:size(imA,2),:);
mask = imresize(frame2im(tim),size(imA(:,:,1)),'nearest');
