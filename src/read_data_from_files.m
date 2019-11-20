clc
close all
clear 

% Read data from text files and sve to mat file. The data can be obtained
% from the ESA Planetary  Science  Archive  or the NASA Planetary Data
% System. For more information see the README.

burst_files = dir('LCV91_OB\RO-C-RPCMAG-3-EXT3-CALIBRATED-V9.0\DATA\CALIBRATED\2016\SEP\LEVEL_C\OB\*M3.TAB');

Px = 0;
Py = 0;
Pz =0 ;

Bx = 0;
By = 0;
Bz = 0;

time=0;

for i=1:length(burst_files)

    filename = horzcat(burst_files(i,:).folder,'\',burst_files(i,:).name);
    delimiter = ' ';

    fprintf('Reading from file: %s\n',filename)

    formatSpec = '%{yyyy-MM-dd''T''HH:mm:ss.SSSSSS}D%*s%f%f%f%f%f%f%*s%[^\n\r]';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string', 'EmptyValue', NaN,  'ReturnOnError', false);
    fclose(fileID);
    ttime = dataArray{:, 1};
    tPx = dataArray{:, 2};
    tPy = dataArray{:, 3};
    tPz = dataArray{:, 4};
    tBx = dataArray{:, 5};
    tBy = dataArray{:, 6};
    tBz = dataArray{:, 7};

    ttime=datenum(ttime);

    clearvars filename delimiter formatSpec fileID dataArray ans;

    Px = vertcat(Px,tPx);
    Py = vertcat(Py,tPy);
    Pz = vertcat(Pz,tPz);


    Bx = vertcat(Bx,tBx);
    By = vertcat(By,tBy);
    Bz = vertcat(Bz,tBz);

    time = vertcat(time,ttime);

end

clearvars ttime i tBx tBy tBz tPx tPy tPz burst_files

