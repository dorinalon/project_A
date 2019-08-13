import numpy as np
import math, random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
#from keras import load_model


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
    print(random_offset)
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

input_dim = 3
output_dim = 1
hidden_size = 30
num_layers = 1

class CustomLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_size
        self.lstm = nn.LSTM(input_size, output_size)
        self.act = nn.Tanh()
        self.linear = nn.Linear(hidden_size, output_size, )


    def forward(self, x):
        pred, hidden = self.lstm(x, None)
        pred = self.act(self.linear(pred)).view(pred.data.shape[0], -1, 1)
        return pred

r= CustomLSTM( hidden_size, input_dim, output_dim)


predictions = []

optimizer = torch.optim.Adam(r.parameters(), lr=1e-2)
loss_func = nn.L1Loss()

running_loss = 0.0
for t in range(301):
    hidden = None
    inp_VR = Variable(torch.Tensor(train_VR.reshape((train_VR.shape[0], -1, 1))), requires_grad=True)
    inp_VS = Variable(torch.Tensor(train_VS.reshape((train_VS.shape[0], -1, 1))), requires_grad=True)
    inp_IL = Variable(torch.Tensor(train_IL.reshape((train_IL.shape[0], -1, 1))), requires_grad=True)

    out = Variable(torch.Tensor(train_QC.reshape((train_QC.shape[0], -1, 1))) )

    pred = r(torch.cat(inp_VR,inp_VS,inp_IL),0)#concat?

    optimizer.zero_grad()
    predictions.append(pred.data.numpy())
    loss = loss_func(pred, out)
    #if t%20==0:
    #    print(t, loss.data[0])
    loss.backward()
    optimizer.step()
    # print statistics
    running_loss = 0.0
    running_loss += loss.item()
    if t%20==0:
        print(t, running_loss)

t_inp_VR = Variable(torch.Tensor(test_VR.reshape((test_VR.shape[0], -1, 1))), requires_grad=False)
t_inp_VS = Variable(torch.Tensor(test_VS.reshape((test_VS.shape[0], -1, 1))), requires_grad=False)
t_inp_IL = Variable(torch.Tensor(test_IL.reshape((test_IL.shape[0], -1, 1))), requires_grad=False)

pred_t = r(torch.cat(t_inp_VR,t_inp_VS,t_inp_IL),0)

# Test loss
runningLossTest = 0.0
lossTest = loss_func(pred_t, Variable(torch.Tensor(test_QC.reshape((test_VR.shape[0], -1, 1)))))#concat?
runningLossTest += lossTest.item()
print(runningLossTest)
