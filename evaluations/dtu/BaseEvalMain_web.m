clear all
close all
format compact %清除空格
clc

% script to calculate distances have been measured for all included scans (UsedSets)

dataPath='G:\dataset\mvsnet\dtu\'; %标准的点云文件，从dtu数据集官网下载
plyPath='D:\Python Programs\MVSNet_pytorch\models\outputs\'; %评估生成的点云文件
resultsPath='D:\Python Programs\MVSNet_pytorch\models\outputs\'; %结果保存的文件

method_string='mvsnet';
light_string='l3'; % l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_'; %results naming
        settings_string='';
end

% get sets used in evaluation
UsedSets=[1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118];

dst=0.2;    %Min dist between points when reducing

for cIdx=1:length(UsedSets)
    %Data set number
    cSet = UsedSets(cIdx)
    %input data name 组合生成的点云ply文件的地址
    DataInName=[plyPath sprintf('%s%03d_%s%s.ply',lower(method_string),cSet,light_string,settings_string)]
    
    %results name mvsnet_Eval_1.mat  
    EvalName=[resultsPath method_string eval_string num2str(cSet) '.mat']
    
    %check if file is already computed 如果mat文件不存在的话
    if(~exist(EvalName,'file'))
        disp(DataInName);
        
        time=clock;time(4:5), drawnow
        
        tic
        Mesh = plyread(DataInName);  %读入点云数据 包含vertex结构体，结构体中有x,y,z,red,green,blue
        Qdata=[Mesh.vertex.x Mesh.vertex.y Mesh.vertex.z]';
        toc
        
        BaseEval=PointCompareMain(cSet,Qdata,dst,dataPath);
        
        disp('Saving results'), drawnow
        toc
        save(EvalName,'BaseEval');
        toc
        
        % write obj-file of evaluation
        % BaseEval2Obj_web(BaseEval,method_string, resultsPath)
        % toc
        time=clock;time(4:5), drawnow
    
        BaseEval.MaxDist=20; %outlier threshold of 20 mm
        
        BaseEval.FilteredDstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
        BaseEval.FilteredDstl=BaseEval.FilteredDstl(BaseEval.FilteredDstl<BaseEval.MaxDist); % discard outliers
    
        BaseEval.FilteredDdata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
        BaseEval.FilteredDdata=BaseEval.FilteredDdata(BaseEval.FilteredDdata<BaseEval.MaxDist); % discard outliers
        
        fprintf("mean/median Data (acc.) %f/%f\n", mean(BaseEval.FilteredDdata), median(BaseEval.FilteredDdata));
        fprintf("mean/median Stl (comp.) %f/%f\n", mean(BaseEval.FilteredDstl), median(BaseEval.FilteredDstl));
    end
end
