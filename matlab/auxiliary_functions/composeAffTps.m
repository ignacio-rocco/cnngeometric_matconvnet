function grAffTps = composeAffTps(grAff,grTps);

hB = size(grTps{1},2);
wB = size(grTps{1},3);

grAffY = squeeze(grAff{1}(1,:,:)); % split into X and Y components
grAffX = squeeze(grAff{1}(2,:,:));

badIdx = grAffY<-1 | grAffY>1 | grAffX<-1 | grAffX>1; % check boundaries
grAffY(badIdx)=nan;
grAffX(badIdx)=nan;

grTpsY = squeeze(grTps{1}(1,:,:)); % split into X and Y components
grTpsX = squeeze(grTps{1}(2,:,:));

[mgX,mgY]=meshgrid(linspace(-1,1,wB),linspace(-1,1,hB));

finalX = interp2(mgX,mgY,grAffX,grTpsX,grTpsY); % compose transformations
finalY = interp2(mgX,mgY,grAffY,grTpsX,grTpsY);

grAffTps = cat(1,permute(finalY,[3,1,2]),permute(finalX,[3,1,2])); % joint final X and Y components