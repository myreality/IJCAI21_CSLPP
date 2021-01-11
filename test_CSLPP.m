clc;
clear;
sampRate=20;
Count=10;

folder_now = pwd;
addpath([folder_now, '\funs']);
datasets_name='Glass';
file=strcat('datasets\',datasets_name,'.mat');
load(file);

[rddata] = CSLPP(X,Y,d,k,alpha,beta,lemmda);
dr_data=real(dr_data);
[mean_measure,std_measure]=cslpp_measure(dr_data,Y,sampRate,Count);
mean_measure
std_measure






%%
function [mean_measure,std_measure]=cslpp_measure(data,target,num_samp,count)
    measure=[];
    for count=1:count
        [~,~,idx_train]=Random_sampling(target,num_samp,'class');
        idx_test=setdiff(1:length(target),idx_train);
        mdl=fitcknn(data(idx_train,:),target(idx_train),'NumNeighbors',1,'Distance','euclidean');
        pred_label=predict(mdl,data(idx_test,:));
        [out] = classification_evaluation(target(idx_test)',pred_label');
        measure=[measure;out.avgAccuracy,out.fscoreMacro,out.fscoreMicro];
    end
    mean_measure=mean(measure,1);
    std_measure=std(measure,1);
end


%%
function [full_label,sample_lbl,sample_lbl_index]=Random_sampling(label,percent,method)
    [n,m]=size(label);
    sample_lbl=[];
    full_label=zeros(n,m)-1;
    switch method
        case 'class'
            [sorted_label,idx_label]=sort(label);
            c=unique(sorted_label)';
            count=0;
            index=[];
            for i=c
                lbl=sorted_label(sorted_label(:,1)==i);
                len=length(lbl);
                num=min(ceil(len*percent/100),len);
                idx=randperm(len,num);
                idx=sort(idx)+count;
                index=[index;idx'];
                count=count+len;
            end
            sample_lbl_index=idx_label(index);
        case 'all'
            num=min(ceil(n*percent/100),n);
            sample_lbl_index=sort(randperm(n,num));
    end   
    for i=sample_lbl_index'
        sample_lbl=[sample_lbl;label(i,:)];
        full_label(i,:)=label(i,:);
    end
    
end