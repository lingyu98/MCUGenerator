"""
Adapted from 
Minimal character-level Vanilla RNN model by Andrej Karpathy (@karpathy)
"""
import argparse
import torch
from utils import parse, batchify, sample


def train(cfg):

    cfg = parse(cfg)
    batch_sl = (cfg.max_T+1)  * cfg.bs_train

    for ep in range(cfg.epochs):
        pp, p, running_loss = 0, 0, 0
        cfg.optimizer.zero_grad()


        while True:
            cfg.model.train()
            
            if p + batch_sl + 1 >= len(cfg.data): break
            seq = [cfg.char_to_ix[ch] for ch in cfg.data[p : p + batch_sl]]
            x, y = batchify(seq, cfg.max_T, cfg.bs_train)

            out = cfg.model(x)
            loss = cfg.loss_fn(out.contiguous().view(cfg.bs_train * cfg.max_T, -1), y.contiguous().view(cfg.bs_train * cfg.max_T))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(cfg.model.parameters(), 0.1)
            cfg.optimizer.step()

            pp += torch.exp(loss)
            p +=  batch_sl
            running_loss += loss.item()

            it = p // batch_sl
            if p % (cfg.log_freq * batch_sl) == 0:
                print('Iter %d| Loss=%.3f' %(it, running_loss))
                sample(cfg)
                running_loss = 0

        print('EP %d| PP=%.2f' %(ep, pp))
        cfg.scheduler.step()




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

    parser.add_argument('--max_T', type=int, default=50)

    parser.add_argument('--bs_train', type=int, default=4, help='training batchsize')
    parser.add_argument('--bs_test', type=int, default=128, help='testing batchsize')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')

    parser.add_argument('--log_freq', type=int, default=10, help='frequency of logging')
    parser.add_argument('--save_freq', type=int, default=100, help='frequency of saving model')
    parser.add_argument('--run_name', type=str, default='test1224')
    parser.add_argument('--datafile', type=str, default='data/final_data.txt')
    # max_T

    configs = parser.parse_args()

    train(configs)
