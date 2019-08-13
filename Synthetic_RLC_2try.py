import numpy as np
import math, random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from numpy import hstack
#matplotlib inline
np.random.seed(80)

# Generating a clean sine wave (actually the freq is 1/60)
def sine(X_, signal_freq=60):
    return np.sin(2 * np.pi * (X_) / signal_freq)

#tmp_sine = sine(200)
#plt.plot(tmp_sine, label = 'sine')
#plt.show()

# Create a noisy and clean sine wave
def sample(sample_size):
    random_offset = random.randint(0, sample_size)
    # print(random_offset)
    X = np.arange(sample_size)
    out = sine(X + random_offset)
    return out

out = sample(100)
plt.plot(out, label ='first sine')
plt.legend()
plt.show()


def create_dataset(n_samples=10000, sample_size=100):
    data_VR = np.zeros((n_samples, sample_size))
    data_IL = np.zeros((n_samples, sample_size))
    data_VS = np.zeros((n_samples, sample_size))


    for i in range(n_samples):
        sample_VR = sample(sample_size)
        data_VR[i, :] = sample_VR
        sample_VS = sample(sample_size)
        data_VS[i, :] = sample_VS
        sample_IL = sample(sample_size)
        data_IL[i, :] = sample_IL
    return data_VR, data_VS, data_IL

data_VR, data_VS, data_IL = create_dataset()
train_VR,train_VS, train_IL = data_VR[:8000], data_VS[:8000], data_IL[:8000]
test_VR, test_VS, test_IL = data_VR[8000:], data_VS[8000:], data_IL[8000:]

C = 1
L = 1
sub_VS_VR = np.subtract(data_VS, data_VR)
#der_IL = (np.gradient(data_IL[0],2))*L
mul_IL = np.zeros((10000, 100))
for i in range(10000):
    mul_IL[i, :] = (data_IL[i] * data_IL[i])*L

data_QC = (np.subtract(sub_VS_VR, mul_IL))*C

train_QC = data_QC[:8000]
test_QC = data_QC[8000:]

train_VR = train_VR.reshape((train_VR.shape[0], -1,1))
train_VS = train_VS.reshape((train_VS.shape[0], -1,1))
train_IL = train_IL.reshape((train_IL.shape[0], -1,1))

TotalTrainInput = torch.cat((train_VR,train_VS, train_IL),0)

# new part
dataset = hstack((data_VR, data_IL, data_VS, data_QC))

class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20 # number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size = n_features,
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first = True)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)


    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)

n_features = 3 # this is number of parallel inputs
n_timesteps = 3 # this is number of timesteps

# convert dataset into input/output
# X, y = split_sequences(dataset, n_timesteps)
# print(X.shape, y.shape)

# create NN
mv_net = MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-1)

batch_size = 16
mv_net.train()
for t in range(8000):
    for b in range(0,len(TotalTrainInput),batch_size):
        inpt = TotalTrainInput[b:b+batch_size,:,:]
        target = train_QC[b:b+batch_size]

        x_batch = torch.tensor(inpt,dtype=torch.float32)
        y_batch = torch.tensor(target,dtype=torch.float32)

        mv_net.init_hidden(x_batch.size(0))
    #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
    #    lstm_out.contiguous().view(x_batch.size(0),-1)
        output = mv_net(x_batch)
        loss = criterion(output.view(-1), y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('step : ' , t , 'loss : ' , loss.item())
