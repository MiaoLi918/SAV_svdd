# Author: Laura Kulowski

import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    # 这个类的主要作用是构建一个LSTM编码器，用于将时间序列数据编码为隐藏状态序列。从 lstm_out 和 self.hidden 中获取编码结果和最终隐藏状态
    def __init__(self, input_size, hidden_size, num_layers = 1):
        
        '''三个参数：
        : param input_size:     the number of features in the input X，输入数据的特征数量
        : param hidden_size:    the number of features in the hidden state h，隐藏状态的特征数量
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)LSTM层数
        '''
        #使用super方法继承类的属性和方法，将传入的参数赋值给类的属性
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer，构建了一个LSTM层，使用nn.LSTM。这个层的参数包括输入特征数、隐藏状态特征数和LSTM的层数。
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)

    def forward(self, x_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)seq_len 是序列的长度，# in batch 表示在一个批次中的样本数量，input_size 是输入特征的数量
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;给出序列中所有隐藏状态
        :hidden gives the hidden state and cell state for the last element in the sequence 给出序列中最后一个元素的隐藏状态和单元状态
        '''
        #batch——size一个批次中的样本数量，是代表一次训练中几个窗口吗？
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        #x_input.shape[0] 表示 x_input 的第一个维度，即时间序列的长度 (seq_len)。x_input.shape[1] 表示 x_input 的第二个维度，即在一个批次中的样本数量 (# in batch)。
        #通过 self.lstm 层将输入数据传递给LSTM层。lstm_out 包含了整个序列中的所有隐藏状态，而 self.hidden 包含了最后一个时间步的隐藏状态和细胞状态。
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        
        '''，初始化隐藏状态，返回一个全零的隐藏状态和细胞状态
        initialize hidden state
        : param batch_size:    x_input.shape[1]。参数 batch_size 表示批次的大小。
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, num_layers = 1):

        '''
        : param input_size:     the number of features in the input X，输入数据的特征数量
        : param hidden_size:    the number of features in the hidden state h ，隐藏状态的特征数量
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs) LSTM层数
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, input_size)            #构建了一个线性层（nn.Linear），用于将LSTM的输出映射到与输入特征数相同的维度。

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D二维 (batch_size, input_size)  #其中 batch_size 是批次的大小，input_size 是输入特征的数量。
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
        output 表示序列中的所有隐藏状态；hidden 表示序列中最后一个元素的隐藏状态和单元状态：
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))       #映射到与输入特征数相同的维度

        '''unsqueeze(0)操作将在第一个维度（批次维度）上添加一个新维度，使其形状变为(1, batch_size, input_size)。这是因为在时间步维度上，x_input 在此时刻只有一个时间步。unsqueeze(0) 的目的是将数据的形状适应LSTM模型的输入要求，即 (seq_len, batch_size, input_size)，其中 seq_len 通常表示时间步。
        squeeze(0) 操作将第一个维度（时间步维度）中的尺寸为1的维度去除，将数据的形状从 (1, batch_size, hidden_size) 变为 (batch_size, hidden_size)。这是为了去除不必要的维度
        '''

        return output, self.hidden


class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_size, hidden_size):

        '''
        : param input_size:     the number of expected features in the input X,输入数据的特征数量
        : param hidden_size:    the number of features in the hidden state h，LSTM层的隐藏状态的特征数量
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)


    def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size, training_prediction = 'recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.01, dynamic_tf = False):
        
        '''
        train lstm encoder-decoder
        训练seq2seq模型，它允许选择不同的训练策略，如递归预测、教师强制或混合教师强制。在每个迭代中，模型接受输入数据和目标数据，并输出预测结果，然后计算损失并执行反向传播。最后，它返回损失数组，以便您可以检查训练过程的损失情况。
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    输入数据形状
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor    输出数据形状
        : param n_epochs:                  number of epochs  迭代数
        : param target_len:                number of values to predict 预测的数值个数
        : param batch_size:                number of samples per gradient update   每次梯度更新的样本数
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'训练预测方法
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch  #动态教师强制 减少每个 epoch 的教师强迫量
        : return losses:                   array of loss function for each epoch   每个迭代损失的函数数组
        '''
        
        # initialize array of losses ，初始化损失数组，数组长度为n_epochs，用于存储每个迭代的损失值。
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()   #定义损失函数均方误差MSE

        # calculate number of batch iterations，计算需要迭代的批次数量 n_batches，将输入数据划分批次进行训练。shape[1] 获取输入数据张量的第二个维度，即批次大小
        n_batches = int(input_tensor.shape[1] / batch_size)
        #例如，如果 input_tensor 的形状为 (100, 64, 10)(seq_len, batch_size, number_features)，并且 batch_size 设置为 16，那么 n_batches 将被计算为 64 / 16 = 4，即有4个批次要用于训练。这意味着在训练中，输入数据将被划分成4个批次，每个批次包含16个样本。
        with trange(n_epochs) as tr:    #trange(n_epochs) 创建了一个 tqdm 迭代器，用于在训练过程中显示进度条
            for it in tr:
                
                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0
                #batch_loss、batch_loss_tf 和 batch_loss_no_tf 分别用于存储当前批次的总损失（根据训练策略不同），以及根据教师强制和递归预测计算的损失。num_tf 和 num_no_tf 用于跟踪使用教师强制和递归预测的样本数量。
                for b in range(n_batches):
                    # select data 选择输入数据批次和目标数据批次。这些批次的形状将是 (seq_len, batch_size, number_features)。
                    input_batch = input_tensor[:, b: b + batch_size, :]
                    target_batch = target_tensor[:, b: b + batch_size, :]

                    # outputs tensor，存储模型的输出。（预测的时间步长，批次大小，输入特征的数量(shape2为第三个维度)）
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                    # initialize hidden state用于存储编码器的中间状态
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient梯度清零，以便进行梯度更新。
                    optimizer.zero_grad()

                    # encoder outputs,获取编码器的输出和隐藏状态
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing，解码器的输入，初始化为输入数据的最后一个时间步
                    decoder_input = input_batch[-1, :, :]   # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden    #初始化为编码器的隐藏状态
                        #根据预测取值，选取不同的训练策略
                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # predict recursively 
                        else:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            
                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]
                            
                            # predict recursively 
                            else:
                                decoder_input = decoder_output

                    # compute the loss ，使用定义的损失函数 criterion 计算模型的输出 outputs 与目标数据 target_batch 之间的损失。计算结果存储在 loss 变量中
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()     #累积变量，用于存储当前迭代中所有批次的总损失
                    
                    # backpropagation，反向传播计算损失梯度，更新模型参数
                    loss.backward()
                    optimizer.step()

                # loss for epoch ，每个迭代循环结束后，计算平均损失值并存储在lossed数组
                batch_loss /= n_batches 
                losses[it] = batch_loss

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02 

                # progress bar ，更新进度条，返回loss数组
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                    
        return losses

    def predict(self, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 预测的目标值数量
        : return np_outputs:       np.array containing predicted values; prediction done recursively ，np.数组，包含预测值；递归预测
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1，增加一维将输入数据的形状变为 (seq_len, 1, input_size)
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions，用于存储模型的预测结果。target_len 是要预测的时间步数，input_tensor.shape[2] 是输入特征的数量。
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]   #初始化解码器的输入 decoder_input 为 input_tensor 的最后一个时间步，形状为 (1, input_size)
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):         #循环递归地进行预测
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output      #更新 decoder_input 为当前时间步的输出，以便在下一个时间步中作为输入使用。
            
        np_outputs = outputs.detach().numpy() #将张量转换为数组，返回预测的值
        
        return np_outputs
