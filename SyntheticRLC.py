import numpy as np
import math, random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
#from keras import load_model

sample_size = 100
#matplotlib inline
np.random.seed(50)

# Generating a clean sine wave
def sine(X_, signal_freq):
    return np.sin(2 * np.pi * (X_) / signal_freq)

# Creating a sine wave with random frequency and random phaze
# The size of the sine vector is sample_size
def sample(sample_size):
    random_offset = random.randint(0, sample_size)
    rand_freq = random.randint(30, 60)
    X = np.arange(sample_size)
    out = sine(X + random_offset,rand_freq)
    return out

# Creating 3 matrices with size of sample_size X n_samples
# each row in every matrix represents a sine wave
def create_dataset(n_samples=10000, sample_size=100):
    data_VR = np.zeros((sample_size, n_samples))
    data_IL = np.zeros((sample_size, n_samples))
    data_VS = np.zeros((sample_size, n_samples))

    for i in range(n_samples):
        sample_VR = sample(sample_size)
        data_VR[:, i] = sample_VR
        sample_VS = sample(sample_size)
        data_VS[:, i] = sample_VS
        sample_IL = sample(sample_size)
        data_IL[:, i] = sample_IL
    return data_VR, data_VS, data_IL

data_VR, data_VS, data_IL = create_dataset()

# Dividing the input dataset into train and test
train_VR,train_VS, train_IL = data_VR[:,:8000], data_VS[:,:8000], data_IL[:,:8000]
test_VR, test_VS, test_IL = data_VR[:,8000:], data_VS[:,8000:], data_IL[:,8000:]


C = 3
L=1

# Creating the output dataset according to a formula
sub_VS_VR = np.subtract(data_VS, data_VR)
mul_IL = np.zeros((sample_size, 10000))
data_QC =  np.zeros((sample_size, 10000))
for i in range(10000):
    L_rand = random.randint(1, 100)
    L_rand = L_rand/100
    mul_IL[:, i] = (data_IL[:,i] * data_IL[:,i])*L

#for i in range(2,100):
 #   data_QC[i,:] = (np.subtract(sub_VS_VR[i-2,:],( mul_IL[i-1,:])))*C
data_QC = (np.subtract(sub_VS_VR,mul_IL))*C

# Dividing the output dataset into train and test
train_QC = data_QC[:,:8000]
test_QC = data_QC[:,8000:]

# Define the nn dimensions
input_dim = 3
output_dim = 1
hidden_size = 40
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

    # Forward function
    def forward(self, x):
        pred, (hidden, context) = self.lstm(x)
        seqLength = x.size(0)
        batchSize = x.size(1)
        out = torch.zeros(seqLength, batchSize, self.output_size)
        for s in range(seqLength):
            out[s] = self.linear(pred[s])
            # out[s] = self.act(self.linear(pred[s]))

        return out

# Creating the lstm nn
r= CustomLSTM( hidden_size, input_dim, output_dim)

predictions = []
optimizer = torch.optim.Adam(r.parameters(), lr=1e-2)
loss_func = nn.L1Loss()

tau = 20 # future estimation time
loss_vec = []
running_loss = 0.0
# TRAIN
for t in range(200):
    hidden = None
    inp_VR = Variable(torch.Tensor(train_VR.reshape((train_VR.shape[0], -1, 1))), requires_grad=True)
    inp_VS = Variable(torch.Tensor(train_VS.reshape((train_VS.shape[0], -1, 1))), requires_grad=True)
    inp_IL = Variable(torch.Tensor(train_IL.reshape((train_IL.shape[0], -1, 1))), requires_grad=True)

    out = Variable(torch.Tensor(train_QC.reshape((train_QC.shape[0], -1, 1))) )
    x = torch.cat((inp_VR,inp_VS,inp_IL),dim=2)

    x = x[:-tau]
    out = out[tau:]

    pred = r(x)
    optimizer.zero_grad()
    predictions.append(pred.data.numpy())
    loss = loss_func(pred, out)

    loss.backward()
    optimizer.step()
    # print statistics
    running_loss = 0.0
    running_loss += loss.item()
    if t%20==0:
        print(t, running_loss)
        plt.plot(pred[:,100].data.numpy(), label='pred[100]')
        plt.plot(out[:, 100].data.numpy(), label='out[100]')
        plt.title(t)
        plt.legend()
        plt.show()

    loss_vec.append(running_loss)

plt.plot(loss_vec)
plt.title("Mean loss")
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

# TEST
t_inp_VR = Variable(torch.Tensor(test_VR.reshape((test_VR.shape[0], -1, 1))), requires_grad=False)
t_inp_VS = Variable(torch.Tensor(test_VS.reshape((test_VS.shape[0], -1, 1))), requires_grad=False)
t_inp_IL = Variable(torch.Tensor(test_IL.reshape((test_IL.shape[0], -1, 1))), requires_grad=False)

x_t = torch.cat((t_inp_VR, t_inp_VS, t_inp_IL), dim=2)
pred_t = r(x_t)

# Test loss
runningLossTest = 0.0
lossTest = loss_func(pred_t, Variable(torch.Tensor(test_QC.reshape((test_QC.shape[0], -1, 1)))))
runningLossTest += lossTest.item()

# plot the mean error for each iteration
plt.plot(pred_t[:,100].data.numpy(), label ='pred_t[100]')
plt.plot(test_QC[tau:,100], label ='TEST_QC[100]')
plt.xlabel('t [sec]')
plt.legend()
plt.show()

# plot the error of a single example
samp_loss_vec = []
for i in range(80):
    a = pred_t[i, 100].data.numpy()
    b = test_QC[i+tau, 100]
    samp_loss_vec.append(abs(np.subtract(a[0],b)))
plt.plot(samp_loss_vec)
plt.title("Error = pred_t[100] - test_Qc[100]")
plt.ylim(0, 10)
plt.xlabel('t [sec]')
plt.show()

print("runningLossTest" + str(runningLossTest))
