function [anno1, in_bound] = TransferAnnotation(anno2, vx,vy)
% x2 = x1 + vx, y2 = y1 + vy

[h1,w1] = size(vx);
[h2,w2] = size(anno2);

[x1,y1] = meshgrid(1:w1, 1:h1);
x2 = x1 + vx;
y2 = y1 + vy;
in_bound = x2 >= 1 & x2 <= w2 & y2 >= 1 & y2 <= h2;

inds1 = sub2ind([h1,w1], y1(in_bound), x1(in_bound));
inds2 = sub2ind([h2,w2], y2(in_bound), x2(in_bound));

anno1 = zeros(h1,w1);
anno1(inds1) = anno2(inds2);







