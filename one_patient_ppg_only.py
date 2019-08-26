import csv
import numpy as np
import math, random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

n_samples = 2
sample_size = 145000
data_ppg = np.zeros((sample_size, n_samples))
data_ecg = np.zeros((sample_size, n_samples))
data_ri = np.zeros((sample_size, n_samples))
data_bp = np.zeros((sample_size, n_samples))

path = '/media/dorin/A4E65F96E65F6796/Project A/one_pers_data/268269-2325-MDC_PRESS_BLD_ART_ABP-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_bp[i,1] = (float(row[1]))
        data_bp[i,0] = (float(row[1]))
        i=i+1
        if(i==145000):
            break


path = '/media/dorin/A4E65F96E65F6796/Project A/one_pers_data/268269-2325-MDC_PULS_OXIM_PLETH-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_ppg[i,1] = (float(row[1]))
        data_ppg[i,0] = (float(row[1]))
        i = i + 1
        if (i == 145000):
            break



path = '/media/dorin/A4E65F96E65F6796/Project A/one_pers_data/268269-2325-MDC_ECG_ELEC_POTL_II-500.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_ecg[i,1] = (float(row[1]))
        data_ecg[i, 0] = (float(row[1]))
        i = i + 1
        if (i == 145000):
            break



path = '/media/dorin/A4E65F96E65F6796/Project A/one_pers_data/268269-2325-MDC_RESP-62.5.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_ri[i,1] = (float(row[1]))
        data_ri[i, 0] = (float(row[1]))
        i = i + 1
        if (i == 145000):
            break


data_bp = data_bp/1500
data_ppg = data_ppg/2500

train_bp = data_bp[:5000,:]
train_ecg = data_ecg[:5000,:]
train_ri = data_ri[:5000,:]
train_ppg = data_ppg[:5000,:]


test_bp = data_bp[5000:6000,:]
test_ecg = data_ecg[5000:6000,:]
test_ri = data_ri[5000:6000,:]
test_ppg = data_ppg[5000:6000,:]

output_dim = 1

#input_dim = 3
input_dim = 1
hidden_size = 30
num_layers = 3

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

        out = torch.zeros(seqLength, batchSize, self.output_size)
        for s in range(seqLength):
            out[s] = self.linear(pred[s])
            # out[s] = self.act(self.linear(pred[s]))

        return out

# Creating the lstm nn
r= CustomLSTM( hidden_size, input_dim, output_dim)

predictions = []
optimizer = torch.optim.Adam(r.parameters(), lr=1e-3)
loss_func = nn.MSELoss()#nn.L1Loss()
#print("here")
tau = 5 # future estimation time
loss_vec = []
running_loss = 0.0
# TRAIN

inp_ppg = Variable(torch.Tensor(train_ppg.reshape((train_ppg.shape[0], -1, 1))), requires_grad=True)
out = Variable(torch.Tensor(train_bp.reshape((train_bp.shape[0], -1, 1))))

inp_ppg = inp_ppg[:1000,0:1,0:1] - 0.8
out = out[:1000,0:1,0:1] - 0.9

for t in range(500):
    hidden = None

    inp_ecg = Variable(torch.Tensor(train_ecg.reshape((train_ecg.shape[0], -1, 1))), requires_grad=True)
    inp_ri = Variable(torch.Tensor(train_ri.reshape((train_ri.shape[0], -1, 1))), requires_grad=True)
   # x = torch.cat((inp_ecg, inp_ri, inp_ppg), dim=2)


   # x = x[:-tau]
   # out = out[tau:]

    #plt.plot(x[:, 1].data.numpy())
    #plt.title("x")
   # plt.show()

    #pred = r(x)
    pred = r(inp_ppg)
    optimizer.zero_grad()
    predictions.append(pred.data.numpy())
    loss = loss_func(pred, out)
    #print("here")

    loss.backward()
    optimizer.step()
    # print statistics
    running_loss = 0.0
    running_loss += loss.item()
    loss_vec.append(running_loss)
    if t%20==0:
        print(t, running_loss)
        plt.subplot(2,1,1)
        plt.plot(inp_ppg[:,0].data.numpy(),label='train_ppg')
        plt.plot(pred[:,0].data.numpy(), label='pred')
        plt.plot(out[:,0].data.numpy(), label='out=train_bp')
        plt.title(t)
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(loss_vec)
        plt.show()



plt.plot(loss_vec)
plt.title("Mean loss")
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

# TEST

t_inp_ecg = Variable(torch.Tensor(test_ecg.reshape((test_ecg.shape[0], -1, 1))), requires_grad=False)
t_inp_ri = Variable(torch.Tensor(test_ri.reshape((test_ri.shape[0], -1, 1))), requires_grad=False)
t_inp_ppg = Variable(torch.Tensor(test_ppg.reshape((test_ppg.shape[0], -1, 1))), requires_grad=False)

x_t = torch.cat((t_inp_ecg,t_inp_ri,t_inp_ppg), dim=2)

#pred_t = r(x_t)
t_inp_ppg = t_inp_ppg - 0.8
test_bp = test_bp - 0.9
pred_t = r(t_inp_ppg)
# Test loss
runningLossTest = 0.0
lossTest = loss_func(pred_t, Variable(torch.Tensor(test_bp.reshape((test_bp.shape[0], -1, 1)))))
runningLossTest += lossTest.item()

# plot the mean error for each iteration
plt.plot(t_inp_ppg[:,1].data.numpy(), label ='test_ppg')
plt.plot(pred_t[:,1].data.numpy(), label ='pred_t')
plt.plot(test_bp[tau:,1], label ='TEST_BP')
plt.xlabel('t [sec]')
plt.legend()
plt.show()

# plot the error of a single example
samp_loss_vec = []
for i in range(80):
    a = pred_t[i,1].data.numpy()
    b = test_bp[i,1]
    samp_loss_vec.append(abs(np.subtract(a[0],b)))
plt.plot(samp_loss_vec)
plt.title("Error = pred_t - test_BP")
plt.ylim(0, 10)
plt.xlabel('t [sec]')
plt.show()

print("runningLossTest" + str(runningLossTest))




