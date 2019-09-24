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
data_bp = np.zeros((sample_size, n_samples))

data_ppg_test_1 = np.zeros((sample_size, n_samples))
data_bp_test_1 = np.zeros((sample_size, n_samples))
data_ppg_test_2 = np.zeros((sample_size, n_samples))
data_bp_test_2 = np.zeros((sample_size, n_samples))
data_ppg_test_3 = np.zeros((sample_size, n_samples))
data_bp_test_3 = np.zeros((sample_size, n_samples))
data_ppg_test_4 = np.zeros((sample_size, n_samples))
data_bp_test_4 = np.zeros((sample_size, n_samples))

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
#####
path = '/media/dorin/A4E65F96E65F6796/Project A/125_patients/2728529-6534-MDC_PRESS_BLD_ART_ABP-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_bp_test_1[i,1] = (float(row[1]))
        data_bp_test_1[i,0] = (float(row[1]))
        i=i+1
        if(i==145000):
            break


path = '/media/dorin/A4E65F96E65F6796/Project A/125_patients/2728529-6534-MDC_PULS_OXIM_PLETH-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_ppg_test_1[i,1] = (float(row[1]))
        data_ppg_test_1[i,0] = (float(row[1]))
        i = i + 1
        if (i == 145000):
            break
##
path = '/media/dorin/A4E65F96E65F6796/Project A/125_patients/2663300-5113-MDC_PRESS_BLD_ART_ABP-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_bp_test_2[i,1] = (float(row[1]))
        data_bp_test_2[i,0] = (float(row[1]))
        i=i+1
        if(i==145000):
            break


path = '/media/dorin/A4E65F96E65F6796/Project A/125_patients/2663300-5113-MDC_PULS_OXIM_PLETH-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_ppg_test_2[i,1] = (float(row[1]))
        data_ppg_test_2[i,0] = (float(row[1]))
        i = i + 1
        if (i == 145000):
            break
##
path = '/media/dorin/A4E65F96E65F6796/Project A/125_patients/2642420-8140-MDC_PRESS_BLD_ART_ABP-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_bp_test_3[i,1] = (float(row[1]))
        data_bp_test_3[i,0] = (float(row[1]))
        i=i+1
        if(i==145000):
            break


path = '/media/dorin/A4E65F96E65F6796/Project A/125_patients/2642420-8140-MDC_PULS_OXIM_PLETH-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_ppg_test_3[i,1] = (float(row[1]))
        data_ppg_test_3[i,0] = (float(row[1]))
        i = i + 1
        if (i == 145000):
            break
##
path = '/media/dorin/A4E65F96E65F6796/Project A/125_patients/2677398-3036-MDC_PRESS_BLD_ART_ABP-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_bp_test_4[i,1] = (float(row[1]))
        data_bp_test_4[i,0] = (float(row[1]))
        i=i+1
        if(i==145000):
            break


path = '/media/dorin/A4E65F96E65F6796/Project A/125_patients/2677398-3036-MDC_PULS_OXIM_PLETH-125.csv'
i=0
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_ppg_test_4[i,1] = (float(row[1]))
        data_ppg_test_4[i,0] = (float(row[1]))
        i = i + 1
        if (i == 145000):
            break


data_bp_test_1 = (data_bp_test_1/2800)-0.8
data_ppg_test_1 = (data_ppg_test_1/2800)-0.8
data_bp_test_2 = (data_bp_test_2/1600)-0.95
data_ppg_test_2 = (data_ppg_test_2/2600)-0.8
data_bp_test_3 = (data_bp_test_3/1900)-0.85
data_ppg_test_3 = (data_ppg_test_3/2600)-0.8
data_bp_test_4 = (data_bp_test_4/2400)-0.85
data_ppg_test_4 = (data_ppg_test_4/2600)-0.8



data_bp = (data_bp/1500) - 0.9
data_ppg = (data_ppg/2500) -0.8



train_bp = data_bp[:5000,:]
train_ppg = data_ppg[:5000,:]


#test_bp = data_bp[5000:6000,:]
#test_ecg = data_ecg[5000:6000,:]
#test_ri = data_ri[5000:6000,:]
#test_ppg = data_ppg[5000:6000,:]

test_bp_1 = data_bp_test_1[20000:22000,:]
test_ppg_1 = data_ppg_test_1[20000:22000,:]

test_bp_2 = data_bp_test_2[20000:22000,:]
test_ppg_2 = data_ppg_test_2[20000:22000,:]

test_bp_3 = data_bp_test_3[20000:22000,:]
test_ppg_3 = data_ppg_test_3[20000:22000,:]

test_bp_4 = data_bp_test_4[20000:22000,:]
test_ppg_4 = data_ppg_test_4[20000:22000,:]

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
tau = 0 # future estimation time
loss_vec = []
running_loss = 0.0
# TRAIN

inp_ppg = Variable(torch.Tensor(train_ppg.reshape((train_ppg.shape[0], -1, 1))), requires_grad=True)
out = Variable(torch.Tensor(train_bp.reshape((train_bp.shape[0], -1, 1))))

#inp_ppg = inp_ppg[:1000,0:1,0:1]
#out = out[:1000,0:1,0:1]

for t in range(400):
    hidden = None



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

        plt.plot(inp_ppg[:1000,0].data.numpy(),label='train_ppg')
        plt.plot(pred[:1000,0].data.numpy(), label='prediction_BP')
        plt.plot(out[:1000,0].data.numpy(), label='train_BP')
        plt.title('iteration '+str(t))
        plt.legend()

        plt.show()



plt.plot(loss_vec)
plt.title("Mean loss")
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

# TEST

t_inp_ppg_1 = Variable(torch.Tensor(test_ppg_1.reshape((test_ppg_1.shape[0], -1, 1))), requires_grad=False)

t_inp_ppg_2 = Variable(torch.Tensor(test_ppg_2.reshape((test_ppg_2.shape[0], -1, 1))), requires_grad=False)

t_inp_ppg_3 = Variable(torch.Tensor(test_ppg_3.reshape((test_ppg_3.shape[0], -1, 1))), requires_grad=False)

t_inp_ppg_4 = Variable(torch.Tensor(test_ppg_4.reshape((test_ppg_4.shape[0], -1, 1))), requires_grad=False)
#x_t = torch.cat((t_inp_ecg,t_inp_ri,t_inp_ppg), dim=2)

#pred_t = r(x_t)


pred_t_1 = r(t_inp_ppg_1)
pred_t_2 = r(t_inp_ppg_2)
pred_t_3 = r(t_inp_ppg_3)
pred_t_4 = r(t_inp_ppg_4)
# Test loss
runningLossTest = 0.0
lossTest = loss_func(pred_t_1, Variable(torch.Tensor(test_bp_1.reshape((test_bp_1.shape[0], -1, 1)))))
runningLossTest += lossTest.item()

# plot the mean error for each iteration
for j in range(1000):
    a = pred_t_1[j,1].data.numpy()
    pred_t_1[j,1] = (((a[0] + 0.8) * 2800) * 0.0625) - 40
test_bp_1[:,1] = ((((test_bp_1[:,1]) + 0.8 ) * 2800) * 0.0625) - 40
plt.plot(pred_t_1[:1000,1].data.numpy(), label ='prediction_BP')
plt.plot(test_bp_1[:1000,1], label ='expected_BP')
plt.xlabel('samples')
plt.ylabel('mmHg')
plt.title("2728529-6534 - BP Prediction based on PPG")
plt.legend()
plt.show()

# plot the mean error for each iteration
for j in range(1000):
    a = pred_t_2[j,1].data.numpy()

    pred_t_2[j,1] = (((a[0] + 0.95) * 1600) * 0.0625) - 40
test_bp_2[:,1] = ((((test_bp_2[:,1]) + 0.95 ) * 1600) * 0.0625) - 40
plt.plot(pred_t_2[:1000,1].data.numpy(), label ='prediction_BP')
plt.plot(test_bp_2[:1000,1], label ='expected_BP')
plt.xlabel('samples')
plt.ylabel('mmHg')
plt.title("2663300-5113 - BP Prediction based on PPG")
plt.legend()
plt.show()

# plot the mean error for each iteration
for j in range(1000):
    a = pred_t_3[j,1].data.numpy()

    pred_t_3[j,1] = (((a[0] + 0.85) * 1900) * 0.0625) - 40
test_bp_3[:,1] = ((((test_bp_3[:,1]) + 0.85 ) * 1900) * 0.0625) - 40
plt.plot(pred_t_3[:1000,1].data.numpy(), label ='prediction_BP')
plt.plot(test_bp_3[:1000,1], label ='expected_BP')
plt.xlabel('samples')
plt.ylabel('mmHg')
plt.title("2642420-8140 - BP Prediction based on PPG")
plt.legend()
plt.show()

# plot the mean error for each iteration
for j in range(1000):
    a = pred_t_4[j,1].data.numpy()

    pred_t_4[j,1] = (((a[0] + 0.85) * 2400) * 0.0625) - 40
test_bp_4[:,1] = ((((test_bp_4[:,1]) + 0.85 ) * 2400) * 0.0625) - 40
plt.plot(pred_t_4[:1000,1].data.numpy(), label ='prediction_BP')
plt.plot(test_bp_4[:1000,1], label ='expected_BP')
plt.xlabel('samples')
plt.ylabel('mmHg')
plt.title("2677398-3036 - BP Prediction based on PPG")
plt.legend()
plt.show()



path = '/media/dorin/A4E65F96E65F6796/Project A/one_pers_data/2728529-6534-expec0_pred1.csv'
i=0
with open(path,'w') as csv_file:
    csv_reader = csv.writer(csv_file, delimiter=',')

    line_count = 0
    for i in range(2000):
        csv_reader.writerow([test_bp_1[i, 1], (pred_t_1[i, 1].data.numpy())[0]])

path = '/media/dorin/A4E65F96E65F6796/Project A/one_pers_data/2663300-5113-expec0_pred1.csv'
i=0
with open(path,'w') as csv_file:
    csv_reader = csv.writer(csv_file, delimiter=',')

    line_count = 0
    for i in range(2000):
        csv_reader.writerow([test_bp_2[i, 1], (pred_t_2[i, 1].data.numpy())[0]])

path = '/media/dorin/A4E65F96E65F6796/Project A/one_pers_data/2642420-8140-expec0_pred1.csv'
i = 0
with open(path, 'w') as csv_file:
    csv_reader = csv.writer(csv_file, delimiter=',')

    line_count = 0
    for i in range(2000):
        csv_reader.writerow([test_bp_3[i, 1], (pred_t_3[i, 1].data.numpy())[0]])

path = '/media/dorin/A4E65F96E65F6796/Project A/one_pers_data/2677398-3036-expec0_pred1.csv'
i = 0
with open(path, 'w') as csv_file:
    csv_reader = csv.writer(csv_file, delimiter=',')

    line_count = 0
    for i in range(2000):
        csv_reader.writerow([test_bp_4[i, 1], (pred_t_4[i, 1].data.numpy())[0]])



# plot the error of a single example
samp_loss_vec = []
for i in range(80):
    a = pred_t_1[i,1].data.numpy()
    b = test_bp_1[i,1]
    samp_loss_vec.append(abs(np.subtract(a[0],b)))
plt.plot(samp_loss_vec)
plt.title("Error = pred_t - test_BP")
plt.ylim(0, 10)
plt.xlabel('t [sec]')
plt.show()

print("runningLossTest" + str(runningLossTest))




