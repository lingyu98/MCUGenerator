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
    cfg.model = GPT(cfg.vocab_size, 192, 6, 6, cfg.max_T)
    cfg.optimizer = torch.optim.AdamW(cfg.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

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
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    return data, data_size, vocab_size, char_to_ix, ix_to_char 

def batchify(seq, max_T, bs):
    '''
    Convert list of tokens into batches of inputs and targets
    seq: list with length max_T + bs
    '''
    torch._assert(len(seq)==max_T + bs, 'sequence length must be larger than max_T x batchsize')
    x = torch.tensor(seq)
    zs = torch.cat([x.roll(shifts=-i, dims=0).unsqueeze(0) for i in range(bs)], 0)
    inputs = zs[:, :max_T]
    targets = zs[:, max_T]

    return inputs, targets
    

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
        pp, p, running_loss = 0, 0, 0
        cfg.optimizer.zero_grad()

        while True:
            cfg.model.train()
            
            if p + cfg.max_T + 1 >= len(cfg.data): break
            seq = [cfg.char_to_ix[ch] for ch in cfg.data[p : p + cfg.max_T+ cfg.bs_train]]
            x, y = batchify(seq, cfg.max_T, cfg.bs_train)

            out = cfg.model.predict_next(x)
            loss = cfg.loss_fn(out, y)

            

            loss.backward()
            torch.nn.utils.clip_grad_norm_(cfg.model.parameters(), 0.1)
            cfg.optimizer.step()

            pp += torch.exp(loss)
            p += cfg.max_T + cfg.bs_train
            running_loss += loss.item()

            it = p // (cfg.max_T + cfg.bs_train)
            if p % (10 * (cfg.max_T + cfg.bs_train)) == 0:
                print('Iter %d| Loss=%.3f' %(it, running_loss))
                running_loss = 0

        print('EP %d| PP=%.2f' %(ep, pp))



        # sample from the model now and then
        # if n % 100 == 0:
        #     sample_ix = sample(hprev, inputs[0], 200)
        #     txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        #     print('---- sample -----')
        #     print('----\n %s \n----' % (txt, ))



        #if n % 100 == 0:
        #    print 'iter %d, loss: %f' % (n, smooth_loss) # print progress





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--normalize', type=bool, default=True, help='normalize inputs')
    parser.add_argument('--log_dir', type=str, default= 'logs/')
    parser.add_argument('--save_dir', type=str, default= 'checkpoints/')
    parser.add_argument('--load_dir', type=str, default='checkpoints/', help='pretrained model path')

    parser.add_argument('--arch', type=str, default='resnet18')

    parser.add_argument('--max_T', type=int, default=100)

    parser.add_argument('--bs_train', type=int, default=16, help='training batchsize')
    parser.add_argument('--bs_test', type=int, default=128, help='testing batchsize')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.1, help='weight decay')

    parser.add_argument('--log_freq', type=int, default=1, help='frequency of logging')
    parser.add_argument('--save_freq', type=int, default=100, help='frequency of saving model')
    parser.add_argument('--run_name', type=str, default='test1224')
    parser.add_argument('--datafile', type=str, default='data/final_data.txt')
    # max_T

    configs = parser.parse_args()

    train(configs)
