A=readtable('268269-2326-MDC_PULS_OXIM_PLETH-125.csv');
x=A{:,1};
y=A{:,2};
figure(1);
%subplot(4,1,1);

plot(x,y);
title('PPG ');
xlabel('time [sec]');
hold all;


B=readtable('268269-2326-MDC_PRESS_BLD_ART_ABP-125.csv');
x=B{:,1};
y=B{:,2};
figure(2);
%subplot(4,1,2);
%y_new = y.*0.0625-40;
plot(x,y);
title('Blood Pressure');
ylabel('Torr');
xlabel('time [sec]');

C=readtable('268269-2326-MDC_RESP-62.5.csv');
x=C{:,1};
y=C{:,2};
figure(3);
%subplot(4,1,3);
plot(x,y);
title('Respiratory Impedance');
xlabel('time [sec]');

D=readtable('911-7931-MDC_ECG_ELEC_POTL_II-500.csv');
x=D{:,1};
y=D{:,2};
figure(4);
%subplot(4,1,4);
plot(x,y);
title('ECG');
xlabel('time [sec]');