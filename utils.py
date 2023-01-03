import torch
from torchtext.data import get_tokenizer
from models.transformer_torch import TransformerModel
from models.transformer import GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse(cfg):
    data, data_size, vocab_size, char_to_ix, ix_to_char = load_data(cfg.datafile)

    cfg.data = data
    cfg.vocab_size = vocab_size
    cfg.char_to_ix = char_to_ix
    cfg.ix_to_char = ix_to_char

    cfg.loss_fn = torch.nn.CrossEntropyLoss()
    cfg.model = GPT(cfg.vocab_size, 768, 12, 12, cfg.max_T).to(device)
    #cfg.model = TransformerModel(cfg.vocab_size, 192, 6, 192, 6, 0.1).to(device)
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
    torch._assert(len(seq)==(max_T+1) * bs, 'sequence length must = max_T + batchsize')
    x = torch.tensor(seq).view(bs, max_T+1)
    inputs, targets = x[:, :-1].to(device), x[:, 1:]

    return inputs, targets
    

def sample(cfg):
    ## a one-hot vector
    print('Sampling ... ')
    p = torch.randint(0, 10000, (1,))
    seq = [cfg.char_to_ix[ch] for ch in cfg.data[p : p + cfg.max_T * cfg.bs_train]]
    x, _ = batchify(seq, cfg.max_T, cfg.bs_train)
    x = x[0:1]

    condition = ' '.join([cfg.ix_to_char[ch] for ch in seq])
    preds = []
    cond = x
    for i in range(10):
        out = cfg.model.predict_next(cond).argmax(1)
        cond = torch.cat([cond[:, 1:], out.unsqueeze(0)], dim=1)
        preds.append(out)
    
    preds = [cfg.ix_to_char[ch.item()] for ch in preds]
    preds = ' '.join(preds)

    print('Condition:', condition)
    print('Preds:', preds)
