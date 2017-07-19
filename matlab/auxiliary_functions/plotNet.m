function [Lxy,Varsxy]=plotNet(net,varargin)
% Function to visualize matconvnet Dag networks
% (to use it with simplenn, convert first with dagnn.loadobj(net))
%
% Ignacio Rocco  <ignacio.rocco-spremolla@inria.fr>
%
% default parameters: 
%
% opts.plotTexts=0;             % detail layer filter size
% fignum=1500;                  % change figure number
% spacingHInputs=12;            % input horiz. spacing
% spacingHLayers=6;             % layer horiz. spacing
% spacingV=1;                   % layer vert. spacing
% textSize=8;                   % standard text size
% 
% Lidx=[];
% Lx = {};              
% Ly = {};                  % custom layer or var. positioning
% Vidx=[];
% Vx = {};
% Vy = {};
%
% boxWidth = 8;
% boxHeight = 1.4;
%
% minimalText = 0;          % trigger "presentation" mode with minimal text
% titleSep=1.25;            %    
% Vtextidx = [];            % indexs of important variables
% Vtextlbl = {};            % override variable labels
% useLatex = 0;             % use Latex to render
% titleSize                 % override title size
%
% To override any parameter, pass comma separated parameter-value pair:
% eg: plotNet(net,'spacingHInputs',8);

opts.plotTexts=0;
opts.fignum=1500;
opts.spacingHInputs=12;
opts.spacingHLayers=6;
opts.spacingV=1;
opts.titleSep=1.25;
opts.textSize=8;
opts.Lxy = [];
opts.Varsxy = [];
opts.Lidx=[];
opts.Lx = [];
opts.Ly = [];
opts.Vidx=[];
opts.Vx = [];
opts.Vy = [];
opts.boxWidth = 8;
opts.boxHeight = 1.4;
opts.minimalText = 0;
opts.Vtextidx = []; % var text idx
opts.Vtextlbl = {}; % override labels
opts.useLatex = 0;
opts.titleSize = round(opts.textSize*2);

opts = vl_argparse(opts, varargin);

figure(opts.fignum); clf; set(gcf,'color','w');
% number of features
Nvars = length(net.vars);
% to store xy coordinates of vars


% check if two inputs exist
varsDepth=zeros(1,Nvars);
for i=1:Nvars
    varsDepth(i)=getNetVarDepth(net.vars(i).name,net);
end
rootIdx = find(varsDepth==0);
rootVarsx=[1:length(rootIdx)]*opts.spacingHInputs-mean([1:length(rootIdx)]*opts.spacingHInputs);

if isempty(opts.Varsxy);
Varsxy = zeros(Nvars,2);
Varsxy(rootIdx,1) = rootVarsx; % shift accordingly in x
else
   userVarsxy = opts.Varsxy;
   Varsxy = zeros(Nvars,2);
   Varsxy(rootIdx,1) = rootVarsx; % shift accordingly in x
end

if ~isempty(opts.Vidx)
    for i=1:length(opts.Vidx)
        if length(opts.Vx)>=i
            if ~isempty(opts.Vx{i})
                Varsxy(opts.Vidx(i),1)=opts.Vx{i};
            end
        end
    end        
    for i=1:length(opts.Vidx)
        if length(opts.Vy)>=i
            if ~isempty(opts.Vy{i})
                Varsxy(opts.Vidx(i),2)=opts.Vy{i};
            end
        end
    end 
end


% number of layers
NL = length(net.layers);

if isempty(opts.Lxy)
    Lxy=zeros(NL,2);
else
    Lxy=opts.Lxy;
end

if ~isempty(opts.Lidx)
    for i=1:length(opts.Lidx)
        if length(opts.Lx)>=i
            if ~isempty(opts.Lx{i})
                Lxy(opts.Lidx(i),1)=opts.Lx{i};
            end
        end
    end        
    for i=1:length(opts.Lidx)
        if length(opts.Ly)>=i
            if ~isempty(opts.Ly{i})
                Lxy(opts.Lidx(i),2)=opts.Ly{i};
            end
        end
    end 
end

for f=1:Nvars
    [Lxy,Varsxy] = positionLayersUsingInput(f, net, Varsxy, Lxy, opts.spacingHLayers, opts.spacingV);
end

if ~isempty(opts.Varsxy)
   Varsxy(userVarsxy~=0)=userVarsxy(userVarsxy~=0);
end

if opts.minimalText==0
    plot(Varsxy(:,1),Varsxy(:,2),'.')
else
    plot(Varsxy(opts.Vtextidx,1),Varsxy(opts.Vtextidx,2),'.')
end
hold on
%plot(Lxy(:,1),Lxy(:,2),'or')
axis equal
%layerLabels = cellstr( [repmat('L',NL,1) num2str([1:NL]')] );

%featureLabels = cellstr( [repmat('x',Nvars,1) num2str([1:Nvars]')] );
%text(Varsxy(:,1), Varsxy(:,2), featureLabels, 'VerticalAlignment','bottom','HorizontalAlignment','right');

for f=1:Nvars
    % feature labels
     if opts.minimalText==0
        if (isempty(net.vars(f).value) || opts.plotTexts==0)
            text(Varsxy(f,1)-0.2, Varsxy(f,2), ['v' num2str(f) ': ' net.vars(f).name], 'VerticalAlignment','middle', 'HorizontalAlignment','right','FontSize',opts.textSize);
        else
            text(Varsxy(f,1)-0.6, Varsxy(f,2), ['v' num2str(f) ': ' net.vars(f).name '(' size2string(size(net.vars(f).value)) ')'], 'VerticalAlignment','middle', 'HorizontalAlignment','right','FontSize',opts.textSize);
        end
     else
         if find(opts.Vtextidx==f)
              if Varsxy(f,2)==0
                 deltaY=-opts.spacingV*opts.titleSep;
             else
                 deltaY=opts.spacingV*opts.titleSep;
              end
             
              if ~isempty(find(opts.Vtextidx==f))
                  if length(opts.Vtextlbl)>=find(opts.Vtextidx==f)
                    if ~isempty(opts.Vtextlbl{find(opts.Vtextidx==f)})
                        lbl = opts.Vtextlbl{find(opts.Vtextidx==f)};
                        formatlabel=0;
                    else
                        formatlabel=1;
                    end
                  else
                      formatlabel=1;
                  end
              else
                 formatlabel=1;
              end
              
              if formatlabel==1
                lbl = net.vars(f).name;
             

             if strcmp(lbl(1:2),'sc')
                 lbl = [lbl(4:end) '^σ^' lbl(3)];
             end
             if length(lbl)>8
                if strcmp(lbl(1:8),'AN1input')
                    lbl = ['I_A' lbl(9:end)];
                end
             end
             if length(lbl)>8
                if strcmp(lbl(1:8),'AN2input')
                    lbl = ['I_B' lbl(9:end)];
                end
             end
            if length(lbl)==8
                if strcmp(lbl(1:8),'AN1input')
                    lbl = ['I_A'];
                elseif strcmp(lbl(1:8),'AN2input')
                    lbl = ['I_B'];
                end
            end             
             if length(lbl)>7
                if strcmp(lbl(1:7),'thetaGt')
                    lbl = ['θ_G_T' lbl(8:end)];
                end
             end
            if length(lbl)==7
                if strcmp(lbl(1:7),'thetaGt')
                    lbl = ['θ_G_T'];
                end
            end
            if length(lbl)>5
                if strcmp(lbl(1:5),'theta')
                    lbl = ['θ_N_N' lbl(6:end)];
                end
             end
             if length(lbl)==5
                if strcmp(lbl(1:5),'theta')
                    lbl = ['θ_N_N'];
                end
             end
            if length(lbl)==4
                if strcmp(lbl(1:4),'loss')
                    lbl = ['L'];
                end
             end
              end
             if opts.useLatex
                text(Varsxy(f,1)-0.2, Varsxy(f,2)+deltaY, [lbl], 'VerticalAlignment','middle', 'HorizontalAlignment','right','FontSize',opts.titleSize,'Interpreter','latex');
             else
                 text(Varsxy(f,1)-0.2, Varsxy(f,2)+deltaY, [lbl], 'VerticalAlignment','middle', 'HorizontalAlignment','right','FontSize',opts.titleSize);
             end
         end
     end
end

                         
% plot lines
for lIdx=1:NL
    iIdx=net.layers(lIdx).inputIndexes;
    for i=1:length(iIdx)
        plot([Varsxy(iIdx(i),1) Lxy(lIdx,1)],[Varsxy(iIdx(i),2) Lxy(lIdx,2)],'k','LineWidth',0.001);
        %quiver(Varsxy(iIdx(i),1),Varsxy(iIdx(i),2),(Lxy(lIdx,1)-Varsxy(iIdx(i),1))/10,(Lxy(lIdx,2)-Varsxy(iIdx(i),2))/10,'k','MarkerSize',20,'AutoScale','off','MaxHeadSize',50)
         headLength=1.5;
         headWidth=1.5;
         ah = annotation('arrow','headStyle','cback1','HeadLength',headLength,'HeadWidth',headWidth);
         set(ah,'parent',gca);
         set(ah,'position',[Varsxy(iIdx(i),1),Varsxy(iIdx(i),2),(Lxy(lIdx,1)-Varsxy(iIdx(i),1))/2,(Lxy(lIdx,2)-Varsxy(iIdx(i),2))/2]);
    end
    oIdx=net.layers(lIdx).outputIndexes;
    plot([Varsxy(oIdx,1) Lxy(lIdx,1)],[Varsxy(oIdx,2) Lxy(lIdx,2)],'k','LineWidth',0.001);
end

% plot layer rectangles
width=opts.boxWidth;
height=opts.boxHeight;
hshift = 10;
for lIdx=1:NL
    if isa(net.layers(lIdx).block,'dagnn.Conv') && strcmp(net.layers(lIdx).name(1:2),'fc')==1
        fillcolor = [255 153 0]/255;
        textcolor = 'w';
        edgecolor = 'none';
    elseif isa(net.layers(lIdx).block,'dagnn.Conv')
        fillcolor = [66 133 234]/255;
        textcolor = 'w';
        edgecolor = 'none';
    elseif isa(net.layers(lIdx).block,'dagnn.ReLU')
        fillcolor = [15 157 88]/255;
        textcolor = 'w';
        edgecolor = 'none';
    elseif isa(net.layers(lIdx).block,'dagnn.Pooling')
        fillcolor = [219 68 55]/255;
        textcolor = 'w';
        edgecolor = 'none';
    else
        fillcolor = [128 128 128]/255;
        textcolor = 'w';
        edgecolor = 'none';
    end
    
    rectangle('Position',[Lxy(lIdx,1)-width/2,Lxy(lIdx,2)-height/2,width,height],'FaceColor',fillcolor,'EdgeColor',edgecolor,...
    'LineWidth',1);    
    if opts.minimalText==0
        text(Lxy(lIdx,1), Lxy(lIdx,2),  ['L' num2str(lIdx) ': ' net.layers(lIdx).name], 'VerticalAlignment','middle','HorizontalAlignment','center','FontSize',opts.textSize,'Color',textcolor);    
    else
        name = net.layers(lIdx).name;
        if length(name)>=4
        if strcmp(name(1:2),'sc')==1
            name=name(4:end);
        end  
        end
        if length(name)>=4
        if strcmp(name(1:3),'AN1')==1 || strcmp(name(1:3),'AN2')==1
            name=name(4:end);
        end
        end
        if length(name)>=3
        if strcmp(name(1:2),'AN')==1 && strcmp(name(1:3),'AND')==0
            name=name(3:end);
        end
        end
        if length(name)>=4
        if strcmp(name(1:3),'_1_')==1 || strcmp(name(1:3),'_2_')==1 || strcmp(name(1:3),'_3_')==1
            name=name(4:end);
        end
        end
        if length(name)>=4
        if strcmp(name(1:4),'Norm')==1
            name='L2norm';
        end
        end
        text(Lxy(lIdx,1), Lxy(lIdx,2),  [name], 'VerticalAlignment','middle','HorizontalAlignment','center','FontSize',opts.textSize,'Color',textcolor);    
    end
    if (isa(net.layers(lIdx).block, 'dagnn.Conv') && opts.plotTexts==1)
        text(Lxy(lIdx,1)-hshift, Lxy(lIdx,2),  ['p' num2str(net.layers(lIdx).paramIndexes(1)) ',p' num2str(net.layers(lIdx).paramIndexes(2)) ': ' size2string(net.layers(lIdx).block.size)], 'VerticalAlignment','middle','HorizontalAlignment','center','FontSize',round(opts.textSize));
        text(Lxy(lIdx,1)-hshift, Lxy(lIdx,2)-0.6,  [' stride: ' size2string(net.layers(lIdx).block.stride)], 'VerticalAlignment','middle','HorizontalAlignment','center','FontSize',round(opts.textSize));
    end
end
% text(Lxy(:,1), Lxy(:,2), layerLabels, 'VerticalAlignment','middle', 'HorizontalAlignment','center');
axis off;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: getNetVarDepth
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function d = getNetVarDepth(varName,anetStruct,d)

if nargin<3
    d=0;
end

if isempty(layersWithOutputVar(varName, anetStruct))
    return
else
    d=d+1;
    layersWithOutputIdx = layersWithOutputVar(varName, anetStruct);
    d=getNetVarDepth(anetStruct.layers(layersWithOutputIdx(1)).inputs{1},anetStruct,d);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: layersWithOutputVar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function layerIdx = layersWithOutputVar(varName, anetStruct)

L = length(anetStruct.layers);
layerIdx=[];
for l=1:L
    if ~isempty(find(strcmp(anetStruct.layers(l).outputs, varName)))
        layerIdx=[layerIdx;l];
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: layersWithInputVar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function layerIdx = layersWithInputVar(varName, anetStruct)

L = length(anetStruct.layers);
layerIdx=[];
for l=1:L
    if ~isempty(find(strcmp(anetStruct.layers(l).inputs, varName)))
        layerIdx=[layerIdx;l];
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: layersUsingInput
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function layerIdx = layersUsingInput(inputIdx, net)

L = length(net.layers);
layerIdx=[];
for l=1:L
    if ~isempty(find(net.layers(l).inputIndexes==inputIdx))
        layerIdx=[layerIdx;l];
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: positionLayersUsingInput
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Lxy,Fxy] = positionLayersUsingInput(inputIdx, net, Fxy, Lxy, spacingH, spacingV)

Fx=Fxy(inputIdx,1);
Fy=Fxy(inputIdx,2);

layerIdx = layersUsingInput(inputIdx, net);

Ly=Fy+spacingV;
Lx=[1:length(layerIdx)]*spacingH-mean([1:length(layerIdx)]*spacingH)+Fx;

% check if layer has multiple inputs
for i=1:length(layerIdx)
    inpIdx=net.layers(layerIdx(i)).inputIndexes;
    if length(inpIdx)==1
        if Lxy(layerIdx(i),1)==0
            Lxy(layerIdx(i),1)=Lx(i);
        end
        if Lxy(layerIdx(i),2)==0
            Lxy(layerIdx(i),2)=Ly;
        end
        % add outputs
        Fidx = net.layers(layerIdx(i)).outputIndexes;
        if Fxy(Fidx,1)==0
            Fxy(Fidx,1)=Lxy(layerIdx(i),1);
        end
        if Fxy(Fidx,2)==0
        Fxy(Fidx,2)=Lxy(layerIdx(i),2)+spacingV;
        end
    elseif max(inpIdx)==inputIdx % do it only once
        if Lxy(layerIdx(i),1)==0
            Lxy(layerIdx(i),1)=mean(Fxy(inpIdx,1));
        end
        if Lxy(layerIdx(i),2)==0
            Lxy(layerIdx(i),2)=max(Fxy(inpIdx,2))+spacingV;
        end
    end
end

% add outputs
for i=1:length(layerIdx)
    inpIdx=net.layers(layerIdx(i)).inputIndexes;
    if max(inpIdx)==inputIdx % do it only once for layers having multiple inputs
        Fidx = net.layers(layerIdx(i)).outputIndexes;
        if Fxy(Fidx,1)==0
            Fxy(Fidx,1)=Lxy(layerIdx(i),1);
        end
        if Fxy(Fidx,2)==0
            Fxy(Fidx,2)=Lxy(layerIdx(i),2)+spacingV;
        end
    end
end
end