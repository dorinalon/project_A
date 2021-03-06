import csv
import os
import numpy as np
import math, random
import statistics as stat
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda")

n_samples = 2
user = 'Ron'

output_dim = 1
input_dim = 1
hidden_size = 16
num_layers = 1
loadPrevious = False

if user == 'Ron':
    only_PPG_path = './data_ppg/'
else:
    only_PPG_path = '/home/shirili/Downloads/ShirirliDorin/project_A/out_data/'

def completeLossFunc(loss_func, pred, out):
    return (torch.mul(loss_func(pred, out), (out - out.mean()).pow(4))).sum()
    #return loss_func(pred, out).sum()

def Train_and_Test(path_bp, path_ppg , patient_num,test1_bp_path,test1_ppg_path,test2_bp_path,test2_ppg_path):

    i = 0
    row_count_bp = 0
    with open(path_bp) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count_bp = sum(1 for line in csv_reader)

    with open(path_bp) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data_bp = np.zeros((row_count_bp, n_samples))
        for row in csv_reader:
            if i == 0:
                print('bp first time-tag %f' % float(row[0]))
            data_bp[i, 1] = (float(row[1]))
            data_bp[i, 0] = (float(row[1]))
            i=i+1


    i = 0
    with open(path_ppg) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count_ppg = sum(1 for line in csv_reader)
    with open(path_ppg) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        data_ppg = np.zeros((row_count_ppg, n_samples))

        for row in csv_reader:
            if i == 0:
                print('ppg first time-tag %f' % float(row[0]))
            data_ppg[i, 1] = (float(row[1]))
            data_ppg[i, 0] = (float(row[1]))
            i = i + 1

    '''
    #test on other patients
    i = 0
    with open(test1_bp_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count_bp_test1 = sum(1 for line in csv_reader)

    with open(test1_bp_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data_bp_test_1 = np.zeros((row_count_bp_test1, n_samples))
        for row in csv_reader:
            data_bp_test_1[i, 1] = (float(row[1]))
            data_bp_test_1[i, 0] = (float(row[1]))
            i = i + 1

    i = 0
    with open(test1_ppg_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count_ppg_test1 = sum(1 for line in csv_reader)

    with open(test1_ppg_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data_ppg_test_1 = np.zeros((row_count_ppg_test1, n_samples))
        for row in csv_reader:
            data_ppg_test_1[i, 1] = (float(row[1]))
            data_ppg_test_1[i, 0] = (float(row[1]))
            i = i + 1
    ##patient2
    i = 0
    with open(test2_bp_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count_bp_test2 = sum(1 for line in csv_reader)

    with open(test2_bp_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data_bp_test_2 = np.zeros((row_count_bp_test2, n_samples))
        for row in csv_reader:
            data_bp_test_2[i, 1] = (float(row[1]))
            data_bp_test_2[i, 0] = (float(row[1]))
            i = i + 1

    i = 0
    with open(test2_ppg_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count_ppg_test2 = sum(1 for line in csv_reader)

    with open(test2_ppg_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data_ppg_test_2 = np.zeros((row_count_ppg_test2, n_samples))
        for row in csv_reader:
            data_ppg_test_2[i, 1] = (float(row[1]))
            data_ppg_test_2[i, 0] = (float(row[1]))
            i = i + 1
    '''
    row_count = min(row_count_ppg,row_count_bp)


    data_bp = data_bp[:row_count]
    data_ppg = data_ppg[:row_count]

    # data normalization
    if True:
        max_bp_div = np.amax(data_bp)
        max_ppg_div = np.amax(data_ppg)

        data_bp = data_bp/max_bp_div
        data_ppg = data_ppg/max_ppg_div

        max_bp = np.amax(data_bp)
        max_ppg = np.amax(data_ppg)
        min_bp = np.amin(data_bp)
        min_ppg = np.amin(data_ppg)

        data_bp = data_bp - ((max_bp - min_bp)/2 + min_bp)
        data_ppg = data_ppg - ((max_ppg - min_ppg)/2 + min_ppg)
    '''
    ##test normalization
    max_bp_test1 = np.amax(data_bp_test_1)
    max_ppg_test1 = np.amax(data_ppg_test_1)

    data_bp_test_1 = data_bp_test_1/max_bp_test1
    data_ppg_test_1 = data_ppg_test_1/max_ppg_test1

    max_bp_test1 = np.amax(data_bp_test_1)
    max_ppg_test1 = np.amax(data_ppg_test_1)
    min_bp_test1 = np.amin(data_bp_test_1)
    min_ppg_test1 = np.amin(data_ppg_test_1)

    data_bp_test_1 = data_bp_test_1 - ((max_bp_test1 - min_bp_test1) / 2 + min_bp_test1)
    data_ppg_test_1 = data_ppg_test_1 - ((max_ppg_test1 - min_ppg_test1) / 2 + min_ppg_test1)
    ##
    max_bp_test2 = np.amax(data_bp_test_2)
    max_ppg_test2 = np.amax(data_ppg_test_2)

    data_bp_test_2 = data_bp_test_2/max_bp_test2
    data_ppg_test_2 = data_ppg_test_2/max_ppg_test2

    max_bp_test2 = np.amax(data_bp_test_2)
    max_ppg_test2 = np.amax(data_ppg_test_2)
    min_bp_test2 = np.amin(data_bp_test_2)
    min_ppg_test2 = np.amin(data_ppg_test_2)

    data_bp_test_2 = data_bp_test_2 - ((max_bp_test2 - min_bp_test2) / 2 + min_bp_test2)
    data_ppg_test_2 = data_ppg_test_2 - ((max_ppg_test2 - min_ppg_test2) / 2 + min_ppg_test2)
    '''
    # division to train and test
    train_size_125Hz =  round(row_count * (5 / 8))
    test_size_125Hz = 2*train_size_125Hz#round(row_count / 4)

    print('train on %d [sec]' % (train_size_125Hz/125))

    train_bp = data_bp[0:train_size_125Hz, :]
    train_ppg = data_ppg[0:train_size_125Hz, :]

    plt.clf()

    #plt.plot(train_ppg[:, 0], label='train_ppg')
    plt.plot(((train_bp[:, 0]  + (((max_bp-min_bp)/2)+min_bp)) * max_bp_div)*0.0625-40, label='train_BP')
    plt.title('train signal @ NN input')
    plt.legend()
    plt.savefig(only_PPG_path + patient_num + '/completeTrainSig')

    test_bp = data_bp[train_size_125Hz:train_size_125Hz+test_size_125Hz, :]
    test_ppg = data_ppg[train_size_125Hz:train_size_125Hz+test_size_125Hz, :]
    '''
    data_bp_test_1 = data_bp_test_1[:test_size_125Hz,:]
    data_ppg_test_1=data_ppg_test_1[:test_size_125Hz,:]
    data_bp_test_2=data_bp_test_2[:test_size_125Hz,:]
    data_ppg_test_2=data_ppg_test_2[:test_size_125Hz,:]
    '''
    # Creating lstm class
    class CustomLSTM(nn.Module):
        # Ctor
        def __init__(self, hidden_size, input_size, output_size, num_layers):
            super(CustomLSTM, self).__init__()
            self.hidden_dim = hidden_size
            self.output_size = output_size
            self.inputNorm = nn.Linear(in_features=input_size, out_features=input_size)
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
            self.act = nn.LeakyReLU()
            self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)
            #self.bn1 = nn.BatchNorm1d(320)
            self.linear01 = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        # Forward function
        def forward(self, x):
            seqLength = x.size(0)
            batchSize = x.size(1)
            y = torch.zeros(x.size()).cuda()
            for s in range(seqLength):
                y[s] = self.inputNorm(x[s])
            pred, (hidden, context) = self.lstm(y)

            out = torch.zeros(seqLength, batchSize, self.output_size).cuda()
            for s in range(seqLength):
                #out[s] = self.linear(self.act(self.linear01(pred[s])))
                out[s] = self.linear(((pred[s])))
                # out[s] = self.act(self.linear(pred[s]))

            return out

    # Creating the lstm nn
    r = CustomLSTM( hidden_size, input_dim, output_dim, num_layers).to(device)
    if loadPrevious:
        r.load_state_dict(torch.load('./ronRun.pt'))
    predictions = []
    optimizer = torch.optim.Adam(r.parameters(), lr=1e-3)
    loss_func = nn.MSELoss(reduction='none').to(device)
    tau = 0 # future estimation time
    loss_vec = []
    running_loss = 0.0

    # TRAIN
    batch_nSamples = 5*60*125
    lastIdx = int((train_ppg[:,0].shape[0])/batch_nSamples)*batch_nSamples
    inp_ppg = torch.Tensor(train_ppg[:lastIdx,0].reshape((batch_nSamples, -1, 1))).cuda()
    out = torch.Tensor(train_bp[:lastIdx,0].reshape((batch_nSamples, -1, 1))).cuda()
    minLoss = 100000.0
    #os.mkdir(only_PPG_path + patient_num )
    for t in range(2000):
        hidden = None
        pred = r(inp_ppg)
        optimizer.zero_grad()
        predictions.append(pred.data.cpu().numpy())
        loss = completeLossFunc(loss_func, pred, out)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss = 0.0
        running_loss += loss.item()
        loss_vec.append(running_loss)
        print(t, running_loss)
        if t%10==0:
            if loss < minLoss:
                minLoss = loss
                torch.save(r.state_dict(), './ronRun.pt')
                print('model saved')

            plt.clf()
            startIdx = int(inp_ppg.shape[0]/2)-500
            stopIdx = int(inp_ppg.shape[0]/2)+500
            plt.plot(inp_ppg[startIdx:stopIdx,0].data.cpu().numpy(),label='train_ppg')
            plt.plot(pred[startIdx:stopIdx,0].data.cpu().numpy(), label='prediction_BP')
            plt.plot(out[startIdx:stopIdx,0].data.cpu().numpy(), label='train_BP')
            plt.title('iteration '+str(t))
            plt.legend()
            plt.savefig(only_PPG_path + patient_num + '/iteration_' + str(t))

    # Mean Loss
    plt.clf()
    plt.plot(loss_vec)
    plt.title("Mean loss")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig(only_PPG_path + patient_num + '/Mean_loss')

    # TEST
    r.load_state_dict(torch.load('./ronRun.pt'))
    r.eval()
    t_inp_ppg = torch.Tensor(test_ppg.reshape((test_ppg.shape[0], -1, 1))).cuda()
    pred_t = r(t_inp_ppg)
    '''
    t_inp_ppg_patient_1 = torch.Tensor(data_ppg_test_1.reshape((data_ppg_test_1.shape[0], -1, 1))).cuda()
    pred_t_patient_1 = r(t_inp_ppg_patient_1)
    t_inp_ppg_patient_2 = torch.Tensor(data_ppg_test_2.reshape((data_ppg_test_2.shape[0], -1, 1))).cuda()
    pred_t_patient_2 = r(t_inp_ppg_patient_2)
    '''
    # Test loss
    runningLossTest = 0.0
    lossTest = completeLossFunc(loss_func, pred_t, (torch.Tensor(test_bp.reshape((test_bp.shape[0], -1, 1)))).cuda())
    runningLossTest += lossTest.item()

    '''
    plt.clf()
    plt.plot(pred_t_patient_1[:1000,1].data.cpu().numpy(), label ='prediction_BP_test_1')
    plt.plot(data_bp_test_1[:1000,1], label ='expected_BP_test_1')
    plt.xlabel('samples')
    plt.title(" BP Prediction based on train on PPG of other patient")
    plt.legend()
    plt.savefig(only_PPG_path + patient_num + '/test_on_patient1')

    plt.clf()
    plt.plot(pred_t_patient_2[:1000,1].data.cpu().numpy(), label ='prediction_BP_test_2')
    plt.plot(data_bp_test_2[:1000,1], label ='expected_BP_test_2')
    plt.xlabel('samples')
    plt.title("BP Prediction based on train on PPG of other patient")
    plt.legend()
    plt.savefig(only_PPG_path + patient_num + '/test_on_patient2')
    '''

    # plot the mean error for each iteration
    for j in range(test_size_125Hz):
        a = pred_t[j,1].data.cpu().numpy()
        pred_t[j,1] = (((a[0] + ((max_bp-min_bp)/2)+min_bp) * max_bp_div) * 0.0625) - 40
    test_bp[:,1] = (((test_bp[:,1] + ((max_bp-min_bp)/2)+min_bp ) * max_bp_div) * 0.0625) - 40
    plt.clf()
    startIdx = int(test_bp.shape[0] / 2) - 500
    stopIdx = int(test_bp.shape[0] / 2) + 500
    plt.plot(pred_t[startIdx:stopIdx,1].data.cpu().numpy(), label ='prediction_BP')
    plt.plot(test_bp[startIdx:stopIdx,1], label ='expected_BP')
    plt.xlabel('samples')
    plt.ylabel('mmHg')
    plt.title("2728529-6534 - BP Prediction based on PPG")
    plt.legend()
    plt.savefig(only_PPG_path + patient_num + '/Prediction and Expected BP')

    '''
    path = only_PPG_path + patient_num + '/expec0_pred1.csv'
    i=0
    with open(path,'w') as csv_file:
        csv_reader = csv.writer(csv_file, delimiter=',')
        line_count = 0
        for i in range(test_size_125Hz):
            csv_reader.writerow([test_bp[i, 1], (pred_t[i, 1].data.cpu().numpy())[0]])


    path = only_PPG_path + patient_num + '/expec0_pred1_test_on_patient1.csv'
    i=0
    with open(path,'w') as csv_file:
        csv_reader = csv.writer(csv_file, delimiter=',')
        line_count = 0
        for i in range(test_size_125Hz):
            csv_reader.writerow([data_bp_test_1[i, 1], (pred_t_patient_1[i, 1].data.cpu().numpy())[0]])


    path = only_PPG_path + patient_num + '/expec0_pred1_test_on_patient2.csv'
    i=0
    with open(path,'w') as csv_file:
        csv_reader = csv.writer(csv_file, delimiter=',')
        line_count = 0
        for i in range(test_size_125Hz):
            csv_reader.writerow([data_bp_test_2[i, 1], (pred_t_patient_2[i, 1].data.cpu().numpy())[0]])

    # plot the error of a single example
    samp_loss_vec = []
    for i in range(test_size_125Hz):
        a = pred_t[i,1].data.cpu().numpy()
        b = test_bp[i,1]
        samp_loss_vec.append(abs(np.subtract(a[0],b)))
    plt.clf()
    plt.plot(samp_loss_vec)
    plt.title("Error = pred_t - test_BP")
    plt.ylim(0, 10)
    plt.xlabel('t [sec]')
    plt.savefig(only_PPG_path + patient_num + '/Error')
    '''

'''
Train_and_Test('/home/shirili/Downloads/ShirirliDorin/project_A/data/2642420-8140-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/2642420-8140-MDC_PULS_OXIM_PLETH-125.csv',
                '2642420-8140',
                '/home/shirili/Downloads/ShirirliDorin/project_A/data/2677398-3036-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/2677398-3036-MDC_PULS_OXIM_PLETH-125.csv',
                '/home/shirili/Downloads/ShirirliDorin/project_A/data/268269-2325-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/268269-2325-MDC_PULS_OXIM_PLETH-125.csv'
               )

Train_and_Test('/home/shirili/Downloads/ShirirliDorin/project_A/data/268269-2325-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/268269-2325-MDC_PULS_OXIM_PLETH-125.csv',
               '268269-2325',
                '/home/shirili/Downloads/ShirirliDorin/project_A/data/2677398-3036-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/2677398-3036-MDC_PULS_OXIM_PLETH-125.csv',
                '/home/shirili/Downloads/ShirirliDorin/project_A/data/2663300-5113-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/2663300-5113-MDC_PULS_OXIM_PLETH-125.csv'
               )

Train_and_Test('/home/shirili/Downloads/ShirirliDorin/project_A/data/2677398-3036-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/2677398-3036-MDC_PULS_OXIM_PLETH-125.csv',
                '2677398-3036',
                '/home/shirili/Downloads/ShirirliDorin/project_A/data/268269-2325-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/268269-2325-MDC_PULS_OXIM_PLETH-125.csv',
                '/home/shirili/Downloads/ShirirliDorin/project_A/data/2642420-8140-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/2642420-8140-MDC_PULS_OXIM_PLETH-125.csv'
               )

Train_and_Test('/home/shirili/Downloads/ShirirliDorin/project_A/data/2663300-5113-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/2663300-5113-MDC_PULS_OXIM_PLETH-125.csv',
                '2663300-5113',
                '/home/shirili/Downloads/ShirirliDorin/project_A/data/2677398-3036-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/2677398-3036-MDC_PULS_OXIM_PLETH-125.csv',
                '/home/shirili/Downloads/ShirirliDorin/project_A/data/2642420-8140-MDC_PRESS_BLD_ART_ABP-125.csv',
               '/home/shirili/Downloads/ShirirliDorin/project_A/data/2642420-8140-MDC_PULS_OXIM_PLETH-125.csv'
               )

Train_and_Test('./data_ppg/268269-2325-MDC_PRESS_BLD_ART_ABP-125.csv',
               './data_ppg/268269-2325-MDC_PULS_OXIM_PLETH-125.csv',
               '268269-2325',
                './data_ppg/2677398-3036-MDC_PRESS_BLD_ART_ABP-125.csv',
               './data_ppg/2677398-3036-MDC_PULS_OXIM_PLETH-125.csv',
                './data_ppg/2663300-5113-MDC_PRESS_BLD_ART_ABP-125.csv',
               './data_ppg/2663300-5113-MDC_PULS_OXIM_PLETH-125.csv'
               )
'''
Train_and_Test('./data_ppg/2677398-3036-MDC_PRESS_BLD_ART_ABP-125.csv',
               './data_ppg/2677398-3036-MDC_PULS_OXIM_PLETH-125.csv',
                '2677398-3036',
                './data_ppg/268269-2325-MDC_PRESS_BLD_ART_ABP-125.csv',
               './data_ppg/268269-2325-MDC_PULS_OXIM_PLETH-125.csv',
                './data_ppg/2642420-8140-MDC_PRESS_BLD_ART_ABP-125.csv',
               './data_ppg/2642420-8140-MDC_PULS_OXIM_PLETH-125.csv'
               )