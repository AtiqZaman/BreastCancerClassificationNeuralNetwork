
%Atiq uz Zaman
%15026391

function accuracy = BreastCancerAccuracy()

       %reading breast canser data file using csvread function
    dataMatr = csvread('BreastCancerData.data');
    
    %weight = load('weight.mat');
    
    
    %finding rows and cols position of zeros in the data file 
    [rows,cols]=find(dataMatr==0);
    
    %calculating mean values of colm 7 
    meanCol7 = round(mean(dataMatr(:,7)));
    
    %replacing every zero in col 7 with mean value
    for i=1:length(rows)
        dataMatr(rows(i),cols(i))=meanCol7;
    end
    
    % training percentage 
    trPercent=0.9;
    
    %testing percentage
    testPercent=0.1;
    
    %seprating benign values from dataset and assigning to benign variable
    benign=dataMatr(dataMatr(:,11)==2,:);
    
    %seprating malignant values from dataset and assigning to benign variable
    malignant=dataMatr(dataMatr(:,11)==4,:);
    
    %Binput = Benign input  
    Binput=benign(:,2:10);
    
    %Minput = Malignent input
    Minput=malignant(:,2:10);
    
    %Boutput = Benign output
    Boutput=benign(:,11);
    Boutput(Boutput==2)=0;
    
    %Moutput = Malignent output
    Moutput=malignant(:,11);
    Moutput(Moutput==4)=1;
                        %--------------------training data ----------------%
    %bnign input data in percentage for training 
    BIpercent=round(length(benign)*trPercent);
    
    %bnign output data in percentage for traning
    BOpercent=round(length(benign)*trPercent);
    
    %malignant input data in percentage for training
    MIpercent=round(length(malignant)*trPercent);
    
    %malignant output data in percentage for training
    MOpercent=round(length(malignant)*trPercent);
    
    %input data: half from from benign and hlaf from Malignant 
    input(1:BIpercent,:)=Binput(1:BIpercent,:);
    input(BIpercent+1:BIpercent+MIpercent,:)=Minput(1:MIpercent,:);
    
    %output data: half from from benign and hlaf from Malignant 
    output(1:BOpercent,:)=Boutput(1:BOpercent,:);
    output(BOpercent+1:BOpercent+MOpercent,:)=Moutput(1:MOpercent,:);
    
                     %--------------------testing data----------------%
    %bnign input data in percentage for testing 
    BIpercentTest=round(length(benign)*testPercent);
    
    %bnign output data in percentage for testing 
    BOpercentTest=round(length(benign)*testPercent);
    
    %Malignant input data in percentage for testing 
    MIpercentTest=round(length(malignant)*testPercent);
    
    %Malignant output data in percentage for testing
    MOpercentTest=round(length(malignant)*testPercent);
    
    %input data for testing: half from from benign and hlaf from Malignant 
    inputTest(1:(BIpercentTest),:)=Binput(BIpercent+1:length(benign),:);
    inputTest(BIpercentTest+1:BIpercentTest+MIpercentTest,:)=Minput(MIpercent+1:length(malignant),:);
    
    %output data for testing: half from from benign and hlaf from Malignant 
    outputTest(1:BOpercentTest,:)=Boutput(BOpercent+1:length(benign),:);
    outputTest(BOpercentTest+1:BOpercentTest+MOpercentTest,:)=Moutput(MOpercent+1:length(malignant),:);
    
    
   
    net = feedforwardnet (13,'trainbr');
    
    %net = setwb(net,weight.weight);
    % Weights = getwb(net);
     
    net.trainParam.epochs = 10;       %epochs to train
    % net.trainParam.mc = 0.004; 
    
    net.trainParam.lr = 0.00001;          %lerning rate
   
    net.trainParam.show = 25;           %Epochs between displays (NaN for no displays)
    net.trainParam.showWindow = true;   %Show training GUI
    net.trainParam.goal = 0.0;            %performance goal
    net.trainParam.max_fail=9;          %Maximum validation failures
    net.trainParam.alpha = 0.001;       %Scale factor that determines sufficient reduction in perf
    net.trainParam.beta = 0.1;          %Scale factor that determines sufficiently large step size
    net.trainParam.delta = 0.01;        %Initial step size in interval location step
    net.trainParam.gama = 0.1;          %Parameter to avoid small reductions in performance, usually set to 0.1 (see srch_cha)
   
    %net = setwb(net,Weights);
    net = setwb(net,rand(55));
    
    
    % training the net using train function
    net = train(net , input' , output');
 
    %simulating the trained net on testing input data
    actualData=round(sim(net,inputTest'));
    
    %Comparing actualData with desired data and assiigning to result
    result=(actualData)'==outputTest;
    testedRows = length(result);
    compRows = find(result==1);
    rowsCount = length(compRows);
    
    % Accuracy formula
    accuracy = (rowsCount/testedRows)*100;


end 