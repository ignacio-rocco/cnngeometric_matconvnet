function thetaVec = paramsToThetaVec(params)

% unwrap params
gamma=params(1); % rotation angle
phi=params(2);   % skew angle
alpha=params(3); % scale along dim1
beta=params(4);  % scale along dim2
tx=params(5);    % translation x
ty=params(6);    % translation y

% convert parameters
Rphi = [cos(phi) -sin(phi); sin(phi) cos(phi)];
Rgamma = [cos(gamma) -sin(gamma); sin(gamma) cos(gamma)];
D=diag([alpha beta]);

A = Rgamma*Rphi'*D*Rphi;

scx = A(1,1); shx=A(1,2); shy=A(2,1); scy=A(2,2);
thetaVec=[scx scy shx shy tx ty];

