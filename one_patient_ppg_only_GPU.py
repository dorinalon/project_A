import csv
import numpy as np
import math, random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import statistics as stat
from torch.autograd import Variable


device = torch.device("cuda")

n_samples = 2
sample_size = 298000
test_len = 5000
train_len = 10000
data_ppg = np.zeros((sample_size, n_samples))
data_bp = np.zeros((sample_size, n_samples))

data_ppg_test_1 = np.zeros((sample_size, n_samples))
data_bp_test_1 = np.zeros((sample_size, n_samples))

path = '/home/shirili/Downloads/ShirirliDorin/project_A/data/268269-2325-MDC_PRESS_BLD_ART_ABP-125.csv'

i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_bp[i,1] = (float(row[1]))
        data_bp[i,0] = (float(row[1]))
        i=i+1
        if(i==298000):
            break

path = '/home/shirili/Downloads/ShirirliDorin/project_A/data/268269-2325-MDC_PULS_OXIM_PLETH-125.csv'

i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_ppg[i,1] = (float(row[1]))
        data_ppg[i,0] = (float(row[1]))
        i = i + 1
        if (i == 298000):
            break
#####

max_bp = np.amax(data_bp)
max_ppg = np.amax(data_ppg)

data_bp = data_bp/max_bp
data_ppg = data_ppg/max_ppg

median_bp = stat.median(data_bp[1])
median_ppg = stat.median(data_ppg[1])

data_bp = data_bp - median_bp
data_ppg = data_ppg - median_ppg

train_bp = data_bp[:train_len,:]
train_ppg = data_ppg[:train_len,:]


test_bp = data_bp[train_len:train_len+test_len,:]
#test_ecg = data_ecg[5000:6000,:]
#test_ri = data_ri[5000:6000,:]
test_ppg = data_ppg[train_len:train_len+test_len,:]

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

        out = torch.zeros(seqLength, batchSize, self.output_size).cuda()
        for s in range(seqLength):
            out[s] = self.linear(pred[s])
            # out[s] = self.act(self.linear(pred[s]))

        return out

# Creating the lstm nn
r= CustomLSTM( hidden_size, input_dim, output_dim).to(device)

predictions = []
optimizer = torch.optim.Adam(r.parameters(), lr=1e-3)
loss_func = nn.MSELoss().to(device)#nn.L1Loss()

tau = 0 # future estimation time
loss_vec = []
running_loss = 0.0
# TRAIN

inp_ppg = torch.Tensor(train_ppg.reshape((train_ppg.shape[0], -1, 1))).cuda()
out = torch.Tensor(train_bp.reshape((train_bp.shape[0], -1, 1))).cuda()

#inp_ppg = inp_ppg[:1000,0:1,0:1]
#out = out[:1000,0:1,0:1]

for t in range(500):
    hidden = None
   # x = x[:-tau]
   # out = out[tau:]

    #plt.plot(x[:, 1].data.numpy())
    #plt.title("x")
   # plt.show()

    #pred = r(x)
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
    if t%30==0:
        #print(t, running_loss)
        plt.clf()
        plt.plot(inp_ppg[round(train_len/2):(round(train_len/2)+1000),0].data.cpu().numpy(),label='train_ppg')
        plt.plot(pred[round(train_len/2):(round(train_len/2)+1000),0].data.cpu().numpy(), label='prediction_BP')
        plt.plot(out[round(train_len/2):(round(train_len/2)+1000),0].data.cpu().numpy(), label='train_BP')
        plt.title('iteration '+str(t))
        plt.legend()

        #plt.show()


plt.clf()
plt.plot(loss_vec)
plt.title("Mean loss")
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()



# TEST
t_inp_ppg = torch.Tensor(test_ppg.reshape((test_ppg.shape[0], -1, 1))).cuda()


#x_t = torch.cat((t_inp_ecg,t_inp_ri,t_inp_ppg), dim=2)

pred_t = r(t_inp_ppg)


# Test loss
runningLossTest = 0.0
lossTest = loss_func(pred_t, (torch.Tensor(test_bp.reshape((test_bp.shape[0], -1, 1)))).cuda())
runningLossTest += lossTest.item()

# plot the mean error for each iteration
for j in range(test_len):
    a = pred_t[j,1].data.cpu().numpy()
    pred_t[j,1] = (((a[0] + median_bp) * max_bp) * 0.0625) - 40
test_bp[:,1] = ((((test_bp[:,1]) + median_bp ) * max_bp) * 0.0625) - 40
plt.plot(pred_t[round(test_len/2):round(test_len/2)+2000,1].data.cpu().numpy(), label ='prediction_BP')
plt.plot(test_bp[round(test_len/2):round(test_len/2)+2000,1], label ='expected_BP')
plt.xlabel('samples')
plt.ylabel('mmHg')
plt.title("2728529-6534 - BP Prediction based on PPG")
plt.legend()
plt.show()


path = '/home/shirili/Downloads/ShirirliDorin/project_A/out_data/2728529-6534-expec0_pred1.csv'
i=0
with open(path,'w') as csv_file:
    csv_reader = csv.writer(csv_file, delimiter=',')

    line_count = 0
    for i in range(test_len):
        csv_reader.writerow([test_bp[i, 1], (pred_t[i, 1].data.cpu().numpy())[0]])


# plot the error of a single example
samp_loss_vec = []
for i in range(test_len):
    a = pred_t[i,1].data.cpu().numpy()
    b = test_bp[i,1]
    samp_loss_vec.append(abs(np.subtract(a[0],b)))
plt.plot(samp_loss_vec)
plt.title("Error = pred_t - test_BP")
plt.ylim(0, 10)
plt.xlabel('t [sec]')
plt.show()

print("runningLossTest" + str(runningLossTest))




