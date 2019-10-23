import csv
import os
import numpy as np
import math, random
import statistics as stat
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
device = torch.device("cuda")

n_samples = 1
output_dim = 1
input_dim = 1
hidden_size = 30
num_layers = 4

only_PPG_path = '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/try_ri_new'
r_path = only_PPG_path + "/saved_nn"

def read_csv(path):
    i = 0
    row_count = 0
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = sum(1 for line in csv_reader)
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = np.zeros((row_count, n_samples))
        for row in csv_reader:
            data[i, 0] = (float(row[0]))
            i=i+1
    return row_count, data

def calc_loss_precentage(expec_BP, pred_BP, time_length, iteration_num, patient_num):
    interval_30_sec = 3700
    expec_95 = []
    expec_5 = []
    pred_95 = []
    pred_5 = []
    for i in range(int(time_length/interval_30_sec) ):
        expec_95.append(np.percentile(expec_BP[i * interval_30_sec:(i + 1) * interval_30_sec], 95))
        expec_5.append(np.percentile(expec_BP[i * interval_30_sec:(i + 1) * interval_30_sec], 5))
        pred_95.append(np.percentile(pred_BP[i * interval_30_sec:(i + 1) * interval_30_sec], 95))
        pred_5.append(np.percentile(pred_BP[i * interval_30_sec:(i + 1) * interval_30_sec], 5))
    plt.clf()
    plt.plot(expec_95, label='expec_95')
    plt.plot(expec_5, label='expec_5')
    plt.plot(pred_95, label='pred_95')
    plt.plot(pred_5, label='pred_5')
    plt.legend()
    plt.title("Prediction and Expected BP systolic and diastolic of "+str(patient_num))
    plt.savefig(only_PPG_path + '/Prediction and Expected BP systolic and diastolic ' + str(iteration_num))

def Train_and_Test_4signals(path_bp, path_ppg, path_ecg, path_ri, patient_num, train_start_time):

    row_count_bp, data_bp = read_csv(path_bp)
    row_count_ppg, data_ppg = read_csv(path_ppg)
    #row_count_ecg, data_ecg = read_csv(path_ecg)
    #row_count_ri, data_ri = read_csv(path_ri)

    #test on other patients

    # row_count_bp_test1, data_bp_test_1 = read_csv(test1_bp_path)
    # row_count_ppg_test1, data_ppg_test_1 = read_csv(test1_ppg_path)
    # row_count_ecg_test1, data_ecg_test_1 = read_csv(test1_ecg_path)
    # row_count_ri_test1, data_ri_test_1 = read_csv(test1_ri_path)

    row_count = min(row_count_ppg,row_count_bp)
    #row_count = min(row_count_ppg, row_count_bp, row_count_ecg, row_count_ri)

    data_bp = data_bp[:row_count]
    data_ppg = data_ppg[:row_count]
    #data_ecg = data_ecg[:row_count]
    #data_ri = data_ri[:row_count]


    train_start_samp = train_start_time*60*125
    train_size_125Hz = 5*60*125
    test_size_125Hz = 5*60*125

    train_bp = data_bp[train_start_samp:train_start_samp+train_size_125Hz, :]
    train_ppg = data_ppg[train_start_samp:train_start_samp+train_size_125Hz, :]
    #train_ecg = data_ecg[train_start_samp:train_start_samp+train_size_125Hz, :]
    #train_ri = data_ri[train_start_samp:train_start_samp+train_size_125Hz, :]

    test_bp = data_bp[train_start_samp+train_size_125Hz:train_start_samp+train_size_125Hz+test_size_125Hz, :]
    test_ppg = data_ppg[train_start_samp+train_size_125Hz:train_start_samp+train_size_125Hz+test_size_125Hz, :]
    #test_ecg = data_ecg[train_start_samp+train_size_125Hz:train_start_samp+train_size_125Hz + test_size_125Hz, :]
    #test_ri = data_ri[train_start_samp+train_size_125Hz:train_start_samp+train_size_125Hz + test_size_125Hz, :]

    test_bp = train_bp
    test_ppg = train_ppg


    # SKlearn min max scaler (normalization)

    scaler_BP = MinMaxScaler(feature_range=(-1,1))
    train_bp_scaled = scaler_BP.fit_transform(train_bp)

    # scaler_ECG = MinMaxScaler(feature_range=(-1,1))
    # train_ecg_scaled = scaler_ECG.fit_transform(train_ecg)
    #
    # scaler_RI = MinMaxScaler(feature_range=(-1,1))
    # train_ri_scaled = scaler_RI.fit_transform(train_ri)

    scaler_PPG = MinMaxScaler(feature_range=(-1,1))
    train_ppg_scaled = scaler_PPG.fit_transform(train_ppg)

    test_bp_scaled = scaler_BP.transform(test_bp)
    # test_ecg_scaled = scaler_ECG.fit_transform(test_ecg)
    # test_ri_scaled = scaler_RI.fit_transform(test_ri)
    test_ppg_scaled = scaler_PPG.fit_transform(test_ppg)




    # Creating lstm class
    class CustomLSTM(nn.Module):
        # Ctor
        def __init__(self, hidden_size, input_size, output_size):
            super(CustomLSTM, self).__init__()
            self.hidden_dim = hidden_size
            self.output_size = output_size
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
            self.act = nn.Tanh()
            self.linear = nn.Linear(in_features=hidden_size, out_features=output_size )
            self.linear01 = nn.Linear(in_features=input_size, out_features=input_size)

        # Forward function
        def forward(self, x):
            seqLength = x.size(0)
            batchSize = x.size(1)
            y = torch.zeros(x.size())
            #for s in range(seqLength):
            #    y[s] = self.linear01(x[s])
            pred, (hidden, context) = self.lstm(x)

            out = torch.zeros(seqLength, batchSize, self.output_size).cuda()
            for s in range(seqLength):
                out[s] = self.linear(pred[s])
                # out[s] = self.act(self.linear(pred[s]))

            return out

    # Creating the lstm nn
    r= CustomLSTM( hidden_size, input_dim, output_dim).to(device)

    predictions = []
    optimizer = torch.optim.Adam(r.parameters(), lr=1e-3)
    loss_func = nn.MSELoss().to(device)
    tau = 0 # future estimation time
    loss_vec = []
    running_loss = 0.0

    # TRAIN
    #inp_ri = torch.Tensor(train_ri_scaled.reshape((train_ri_scaled.shape[0], -1, 1))).cuda()
    inp_ppg = torch.Tensor(train_ppg_scaled.reshape((train_ppg_scaled.shape[0], -1, 1))).cuda()
    #inp_ecg = torch.Tensor(train_ecg_scaled.reshape((train_ecg_scaled.shape[0], -1, 1))).cuda()
    out = torch.Tensor(train_bp_scaled.reshape((train_bp_scaled.shape[0], -1, 1))).cuda()

    # TEST

    x_test_PPG = torch.Tensor(test_ppg_scaled.reshape((test_ppg_scaled.shape[0], -1, 1))).cuda()
    #x_test_ri = torch.Tensor(test_ri_scaled.reshape((test_ri_scaled.shape[0], -1, 1))).cuda()
    #x_test_ecg = torch.Tensor(test_ecg_scaled.reshape((test_ecg_scaled.shape[0], -1, 1))).cuda()

    pred_t = r(x_test_PPG)

    for t in range(321):
        hidden = None
        #x = torch.cat((inp_ppg, inp_ecg, inp_ri), dim=2)
        pred = r(inp_ppg)
        optimizer.zero_grad()
        predictions.append(pred.data.cpu().numpy())
        loss = loss_func(pred, out)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss = 0.0
        running_loss += loss.item()
        loss_vec.append(running_loss)
        print(t, running_loss)
        if t%20==0:

            plt.clf()
            # plt.plot(inp_ppg[:(5*60*125),0].data.cpu().numpy(),label='train_ppg')
            # plt.plot(inp_ecg[:1000, 0].data.cpu().numpy(), label='train_ecg')
            # plt.plot(inp_ri[:1000, 0].data.cpu().numpy(), label='train_ri')
            plt.plot(pred[:(5*60*125),0].data.cpu().numpy(), label='prediction_BP')
            plt.plot(out[:(5*60*125),0].data.cpu().numpy(), label='train_BP')
            plt.title('iteration '+str(t))
            plt.legend()
            plt.savefig(only_PPG_path + '/iteration_' + str(t) +" from minute "+ str(train_start_time))
            calc_loss_precentage(out[:, 0].data.cpu(), pred[:, 0].data.cpu().numpy(), train_size_125Hz,
                                 str(t) + 'from minute' + str(train_start_time), patient_num)
            pred_t = r(x_test_PPG)

            plt.clf()
            plt.plot(pred_t[:1000, 0].data.cpu().numpy(), label='prediction_BP')
            plt.plot(test_bp_scaled[:1000, 0], label='expected_BP')
            plt.xlabel('samples')
            plt.ylabel('mmHg')
            plt.title(" BP Prediction based on PPG iteration " + str(t))
            plt.legend()
            plt.savefig(only_PPG_path + '/Prediction and Expected BP ' + str(t)+" from minute "+str(train_start_time))

    # Save the net
    torch.save(r.state_dict(),r_path )
    # Mean Loss
    plt.clf()
    plt.plot(loss_vec)
    plt.title("Mean loss")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig(only_PPG_path + '/Mean_loss'+" from minute "+ str(train_start_time))

    # TEST- on the same patient as the train

    pred_t = r(x_test_PPG)

    # TEST- other patient
    # t_inp_ppg_patient_1 = torch.Tensor(data_ppg_test_1.reshape((data_ppg_test_1.shape[0], -1, 1))).cuda()
    # t_inp_ecg_patient_1 = torch.Tensor(data_ecg_test_1.reshape((data_ecg_test_1.shape[0], -1, 1))).cuda()
    # t_inp_ri_patient_1 = torch.Tensor(data_ri_test_1.reshape((data_ri_test_1.shape[0], -1, 1))).cuda()
    # x_t_patient_1 = torch.cat((t_inp_ppg_patient_1, t_inp_ecg_patient_1, t_inp_ri_patient_1), dim=2)
    # pred_t_patient_1 = r(x_t_patient_1)

    # Test loss
    runningLossTest = 0.0
    lossTest = loss_func(pred_t, (torch.Tensor(test_bp.reshape((test_bp.shape[0], -1, 1)))).cuda())
    runningLossTest += lossTest.item()

    # plt.clf()
    # plt.plot(pred_t_patient_1[:1000,1].data.cpu().numpy(), label ='prediction_BP_test_1')
    # plt.plot(data_bp_test_1[:1000,1], label ='expected_BP_test_1')
    # plt.xlabel('samples')
    # plt.title(" BP Prediction based on train on PPG of other patient")
    # plt.legend()
    # plt.savefig(only_PPG_path + patient_num + '/test_on_patient1')


    pred_t = scaler_BP.inverse_transform(pred_t[:, :,0].data.cpu().numpy())
    test_bp_scaled = scaler_BP.inverse_transform(test_bp_scaled)[:,:]

    test_bp_scaled = (test_bp_scaled * 0.0625) - 40



    plt.clf()
    plt.plot(pred_t[:1000,0], label ='prediction_BP')
    plt.plot(test_bp[:1000,0], label ='expected_BP')
    plt.xlabel('samples')
    plt.ylabel('mmHg')
    plt.ylabel('mmHg')
    plt.title(" BP Prediction based on PPG - original values")
    plt.legend()
    plt.savefig(only_PPG_path + '/FINAL ZOOM: Prediction and Expected BP from minute '+str(train_start_time))

    plt.clf()
    plt.plot(pred_t[:,0], label='prediction_BP')
    plt.plot(test_bp[:,0], label='expected_BP')
    plt.xlabel('samples')
    plt.ylabel('mmHg')
    plt.ylabel('mmHg')
    plt.title(" BP Prediction based on PPG - original values")
    plt.legend()
    plt.savefig(only_PPG_path + '/FINAL: Prediction and Expected BP from minute ' + str(train_start_time))
    plt.clf()

    calc_loss_precentage(test_bp[:, 0], pred_t[:, 0], test_size_125Hz, 'orig value from minute '+str(train_start_time), patient_num)
    path = only_PPG_path + '/from minute_'+ str(train_start_time)+'_expec0_pred1.csv'
    i=0
    with open(path,'w') as csv_file:
        csv_reader = csv.writer(csv_file, delimiter=',')
        line_count = 0
        for i in range(test_size_125Hz):
            csv_reader.writerow([test_bp[i, 0], pred_t[i, 0]])


    # path = only_PPG_path + patient_num + '/expec0_pred1_test_on_patient1.csv'
    # i=0
    # with open(path,'w') as csv_file:
    #     csv_reader = csv.writer(csv_file, delimiter=',')
    #     line_count = 0
    #     for i in range(test_size_125Hz):
    #         csv_reader.writerow([data_bp_test_1[i, 1], (pred_t_patient_1[i, 1].data.cpu().numpy())[0]])





# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2210563-9215/2210563-9215_BP_125Hz.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2210563-9215/2210563-9215_PPG_125Hz.csv',
#                         '2210563-9215',0)
#
# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2210563-9215/2210563-9215_BP_125Hz.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2210563-9215/2210563-9215_PPG_125Hz.csv',
#                         '2210563-9215',15)
#
# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/268269-2325/268269-2325-MDC_PRESS_BLD_ART_ABP-125.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/268269-2325/268269-2325-MDC_PULS_OXIM_PLETH-125.csv',
#                         '268269-2325',0)
#
# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/268269-2325/268269-2325-MDC_PRESS_BLD_ART_ABP-125.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/268269-2325/268269-2325-MDC_PULS_OXIM_PLETH-125.csv',
#                         '268269-2325',15)
#
#
# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2642420-8086/2642420-8086_BP_125Hz.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2642420-8086/2642420-8086_PPG_125Hz.csv',
#                         '2642420-8086',0)
#
# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2642420-8086/2642420-8086_BP_125Hz.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2642420-8086/2642420-8086_PPG_125Hz.csv',
#                         '2642420-8086',15)
#
#
#
#
# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2728529-6534/2728529-6534_BP_125Hz.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2728529-6534/2728529-6534_PPG_125Hz.csv',
#                         '2728529-6534',0)

# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2728529-6534/2728529-6534_BP_125Hz.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2728529-6534/2728529-6534_PPG_125Hz.csv',
#                         '2728529-6534',10)
#
# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2544444-5692/2544444-5692_BP_125Hz.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_4_sig/2544444-5692/2544444-5692_PPG_125Hz.csv',
#                        '2544444-5692',0
#                         )
#
# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534BP_125Hz_1_col.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534PPG_125Hz_1_col.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534ECG_125Hz_1_col.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534RI_125Hz_1_col.csv',
#                        '2728529-6534',12)


Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534BP_125Hz_1_col.csv',
                        '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534PPG_125Hz_1_col.csv',
                        '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534PPG_125Hz_1_col.csv',
                        '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534PPG_125Hz_1_col.csv',
                       '2728529-6534',12)

# Train_and_Test_4signals('/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2642420-8079/2642420-8079-MDC_PRESS_BLD_ART_ABP-125.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2642420-8079/2642420-8079-MDC_PULS_OXIM_PLETH-125.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534ECG_125Hz_1_col.csv',
#                         '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2728529-6534/2728529-6534RI_125Hz_1_col.csv',
#                        '2642420-8079',5)

