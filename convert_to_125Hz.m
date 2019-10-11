close all;

%% fill in this :
person_path = ('C:\\Users\\dorin\\Documents\\semester_6\\projectA\\matlab_trying\\sec_patient_transferTo500\\2728529-6534');
person_name = '2728529-6534';

%% create graphs
BP_str = '-MDC_PRESS_BLD_ART_ABP-125.csv';
PPG_str = '-MDC_PULS_OXIM_PLETH-125.csv'; 
RI_str = '-MDC_RESP-62.5.csv';
ECG_str = '-MDC_ECG_ELEC_POTL_II-500.csv';

BP_filename = strcat(person_path, '\\', person_name, BP_str);
PPG_filename = strcat(person_path, '\\', person_name, PPG_str);
RI_filename = strcat(person_path, '\\', person_name, RI_str);
ECG_filename = strcat(person_path, '\\', person_name, ECG_str);

%% BP graph
BP_table = readtable(BP_filename);
x_BP=BP_table{:,1};
y_BP=BP_table{:,2};
z_BP=BP_table{1,1};
x_BP=x_BP-z_BP;
y_BP_new = y_BP.*0.0625-40;
figure(1);
hold on
plot(x_BP,y_BP_new);
title('Blood Pressure');
ylabel('mmHg');
xlabel('time [sec]');
hold all;

%% PPG graph
PPG_table = readtable(PPG_filename);
x_PPG=PPG_table{:,1};
y_PPG=PPG_table{:,2};
z_ppg=PPG_table{1,1};
x_PPG=x_PPG-z_ppg;
figure(2);
hold on
plot(x_PPG,y_PPG);
title('PPG');
xlabel('time [sec]');
hold all;

%% RI
RI_table = readtable(RI_filename);
x_RI = RI_table{:,1};
y_RI = RI_table{:,2};
z_RI = RI_table{1,1};
x_RI = x_RI - z_RI;
figure(3);
plot(x_RI, y_RI);
title('Respiratory Impedance');
xlabel('time [sec]');

%% ECG
ECG_table = readtable(ECG_filename);
x_ECG = ECG_table{:,1};
y_ECG = ECG_table{:,2};
z_ECG = ECG_table{1,1};
x_ECG = x_ECG-z_ECG;
figure(4);
plot(x_ECG,y_ECG);
title('ECG');
xlabel('time [sec]');
% if (length(y_ecg) > 1000000) %% what does it do???
%     ecg_500Hz = y_ecg(1:1000000);
% end
% plot(x_ecg,y_ecg);
% title('ECG 500 Hz');

%% sampling ECG to 125 Hz

f = 500;
T =1/f;
y_ECG_125 = y_ECG(1:4:end);
x_ECG_125 = 1:length(y_ECG_125);
figure(5)
plot(x_ECG_125,y_ECG_125);
title('ECG 125 Hz');

%% interpulation (RI)
y_ri_125Hz = interp(y_RI, 2);
figure(6)
plot(y_ri_125Hz);
title('RI 125 Hz');

%% make csv's
% RI
OutputFilePath = person_path;
OutputFileName = 'RI_125Hz_1_col.csv';
filename = fullfile(OutputFilePath, OutputFileName);
[fid, msg] = fopen(filename, 'wt');
if fid < 0
  error('Could not open file "%s" because "%s"', fid, msg);
end
for k=1:size(y_ri_125Hz)
   fprintf(fid,'%f\n',y_ri_125Hz(k));
end
fclose(fid);

% BP
OutputFileName = 'BP_125Hz_1_col.csv';
filename = fullfile(OutputFilePath, OutputFileName);
[fid, msg] = fopen(filename, 'wt');
if fid < 0
  error('Could not open file "%s" because "%s"', fid, msg);
end
for k=1:size(y_BP_new)
   fprintf(fid,'%f\n',y_BP_new(k));
end
fclose(fid);

% PPG
OutputFileName = 'PPG_125Hz_1_col.csv';
filename = fullfile(OutputFilePath, OutputFileName);
[fid, msg] = fopen(filename, 'wt');
if fid < 0
  error('Could not open file "%s" because "%s"', fid, msg);
end
for k=1:size(y_PPG)
   fprintf(fid,'%f\n',y_PPG(k));
end
fclose(fid);

% ECG
OutputFileName = 'ECG_125Hz_1_col.csv';
filename = fullfile(OutputFilePath, OutputFileName);
[fid, msg] = fopen(filename, 'wt');
if fid < 0
  error('Could not open file "%s" because "%s"', fid, msg);
end
for k=1:size(y_ECG_125)
   fprintf(fid,'%f\n',y_ECG_125(k));
end
fclose(fid);