"""
Adapted from 
Minimal character-level Vanilla RNN model by Andrej Karpathy (@karpathy)
"""
from torchtext.data import get_tokenizer
import argparse
import re
import numpy as np
import torch
from models.transformer import GPT


def parse(cfg):
    data, data_size, vocab_size, char_to_ix, ix_to_char = load_data(configs.datafile)

    cfg.data = data
    cfg.vocab_size = vocab_size
    cfg.char_to_ix = char_to_ix
    cfg.ix_to_char = ix_to_char

    cfg.loss_fn = torch.nn.CrossEntropyLoss()
    cfg.model = GPT(cfg.vocab_size, 128, 4, 4, cfg.max_T)

    return cfg

def load_data(datafile):
    #data = open(datafile, 'r').read() # should be simple plain text file
    tokenizer = get_tokenizer("basic_english")
    with open(datafile) as file:
        data = tokenizer(file.read())


    # use set() to count the vocab size
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d words, %d unique.' % (data_size, vocab_size))

    # dictionary to convert char to idx, idx to char
    char_to_ix = { ch:i+1 for i,ch in enumerate(chars) }
    ix_to_char = { i+1:ch for i,ch in enumerate(chars) }

    return data, data_size, vocab_size, char_to_ix, ix_to_char 

# def sample(configs):
#     ## a one-hot vector
#     x = torch.zeros((configs.vocab_size, 1))
#     x[seed_ix] = 1

#     ixes = []
#     for t in range(n):
#         ## self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
#         h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
#         ## y = np.dot(self.W_hy, self.h)
#         y = np.dot(Why, h) + by
#         ## softmax
#         p = np.exp(y) / np.sum(np.exp(y))
#         ## sample according to probability distribution
#         ix = np.random.choice(range(vocab_size), p=p.ravel())

#         ## update input x
#         ## use the new sampled result as last input, then predict next char again.
#         x = np.zeros((vocab_size, 1))
#         x[ix] = 1

#         ixes.append(ix)

#     return ixes

def train(cfg):

    cfg = parse(cfg)
    for ep in range(cfg.epochs):
        p = 0
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + cfg.max_T + 1 >= len(cfg.data): pass

        inputs = [cfg.char_to_ix[ch] for ch in cfg.data[p : p + cfg.max_T]]
        targets = [cfg.char_to_ix[ch] for ch in cfg.data[p + 1 : p + cfg.max_T + 1]]

        # sample from the model now and then
        # if n % 100 == 0:
        #     sample_ix = sample(hprev, inputs[0], 200)
        #     txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        #     print('---- sample -----')
        #     print('----\n %s \n----' % (txt, ))
        print(inputs)
        print(targets)
        return 0


        #if n % 100 == 0:
        #    print 'iter %d, loss: %f' % (n, smooth_loss) # print progress


        p += cfg.max_T # move data pointer
        n += 1 # iteration counter 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--normalize', type=bool, default=True, help='normalize inputs')
    parser.add_argument('--log_dir', type=str, default= 'logs/')
    parser.add_argument('--save_dir', type=str, default= 'checkpoints/')
    parser.add_argument('--load_dir', type=str, default='checkpoints/', help='pretrained model path')

    parser.add_argument('--arch', type=str, default='resnet18')

    parser.add_argument('--max_T', type=int, default=25)

    parser.add_argument('--bs_train', type=int, default=128, help='training batchsize')
    parser.add_argument('--bs_test', type=int, default=128, help='testing batchsize')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--log_freq', type=int, default=1, help='frequency of logging')
    parser.add_argument('--save_freq', type=int, default=100, help='frequency of saving model')
    parser.add_argument('--run_name', type=str, default='test1224')
    parser.add_argument('--datafile', type=str, default='data/final_data.txt')
    # max_T

    configs = parser.parse_args()

    train(configs)
