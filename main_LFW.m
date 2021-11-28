clear; clc; close all
c=158;% The class number of training samples


imgsize_m=32;% image size
imgsize_n=32;

iterations=10;  %iteration number

% repeat the experiment ten times
for train_num=5
for lambda=[0.01] %[100 10 1 0.1 0.01 0.001 0.0001 0.00001]
    for iter=1:iterations
        [train_data3,train_label,test_data,test_label]=readdata(train_num,c); % input data
        
        % convert the trainning set into different resolutions
        [train_data3]=changedatasize(train_data3,imgsize_m,imgsize_n,imgsize_m,imgsize_n);
        [res_m,res_n,train_data2] = changeResolution(train_data3,0,imgsize_m,imgsize_n);
        [res_m,res_n,train_data2] = changeResolution(train_data2,1,res_m,res_n);
        [train_data2]=changedatasize(train_data2,res_m,res_n,imgsize_m,imgsize_n);
        [res_m,res_n,train_data4] = changeResolution(train_data3,0,imgsize_m,imgsize_n);
        [res_m,res_n,train_data4] = changeResolution(train_data4,0,res_m,res_n);
        [res_m,res_n,train_data4] = changeResolution(train_data4,1,res_m,res_n);
        [res_m,res_n,train_data4] = changeResolution(train_data4,1,res_m,res_n);
        train_data4=changedatasize(train_data4,res_m,res_n,imgsize_m,imgsize_n);
        
        % convert the test set into different resolutions
        test_data=mixtestdata(test_data,test_label,imgsize_m,imgsize_n);
        
         train_data2=train_data2';
         train_data3=train_data3';
         train_data4=train_data4';
        test_data=test_data';
        test_label=test_label';
        train_label=train_label';
        
        A = [train_data3,train_data2,train_data4];
        M = eye(train_num*c*3);
        Temp1 = inv(A'*A+lambda*M)*A';
        n = train_num*c;
        Nt=size(test_data,2);
 
        ID=[];
         tic
        for indTest = 1:size(test_data,2)
            [id]    =   Classification_MRCR(train_data3,train_data2,train_data4,Temp1,test_data(:,indTest),train_label,n);
            ID      =   [ID id];
        end
        cornum      =   sum(ID==test_label);
        accracy =   [cornum/length(test_label)];

         time_MRDCR = toc/Nt; 
       % fprintf(['train_num=' num2str(train_num) ' lambda=' num2str(lambda) ' accracy=' num2str(accracy)  '\n']);
         fprintf(['train_num=' num2str(train_num) ' lambda=' num2str(lambda) ' accracy=' num2str(accracy) ' time=' num2str(time_MRDCR) '\n']);
        acc(iter,1) = accracy;
        time(iter,1)=time_MRDCR;
        iter = iter + 1 ;
    end
    Ravg = mean(acc);
    Rstd = std(acc);
    avg_time=mean(time);
    fprintf(['train_num=' num2str(train_num) ' lambda=' num2str(lambda) ' Ravg=' num2str(Ravg)  ' Rstd=' num2str(Rstd)  ' avg_time=' num2str(avg_time) '\n']);  
end
end
