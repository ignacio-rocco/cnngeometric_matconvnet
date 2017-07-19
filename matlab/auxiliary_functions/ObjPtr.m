%% Function taken from 
%  "Deformable Spatial Pyramid Matching for Fast Dense Correspondences"
%
%   Jaechul Kim1, Ce Liu2, Fei Sha3, Kristen Grauman1
%
%  http://vision.cs.utexas.edu/projects/dsp/
%
function [x1 y1] = ObjPtr(anno)
[y1 x1] = find(anno);
lx1 = min(x1);
rx1 = max(x1);
ty1 = min(y1);
dy1 = max(y1);
w1 = rx1-lx1 + 1;
h1 = dy1-ty1 + 1;
[x y] = meshgrid(1:size(anno,2), 1:size(anno,1));
x1 = (x - lx1)./w1;
y1 = (y - ty1)./h1;
end
