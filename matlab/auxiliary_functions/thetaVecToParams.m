function params = thetaVecToParams(thetaVec)

% convert parameters
A = [thetaVec(1) thetaVec(3); thetaVec(4) thetaVec(2)];
[U,S,V]=svd(A);
Rtheta = U*V'; % rotation angle
Rphi = V';     % shear angle
rot_angle = atan2(Rtheta(2,1),Rtheta(2,2));
sh_angle = atan2(Rphi(2,1),Rphi(2,2));
lambda_1=S(1,1);
lambda_2=S(2,2);
tx = thetaVec(5);
ty = thetaVec(6);

params = [rot_angle, sh_angle, lambda_1, lambda_2, tx, ty];
