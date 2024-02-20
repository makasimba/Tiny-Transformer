import numpy as np
import math
import torch
from model import GPTModel
from tqdm import tqdm
import random
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DIR = 'Models'

def w_and_b(sh_of_w):
    """ glorot w """
    a, b = sh_of_w
    v = math.sqrt(6/(a+b))
    w = np.random.uniform(low=-v, high=v, size=sh_of_w)
    return w

file_path = 'names.txt'
data = open(file_path, 'r').readlines()
data = list(set(data))
random.shuffle(data)
data = ''.join(data)


v = sorted(list(set(data)))
n_v = len(v)

encoder = {ch:i for i, ch in enumerate(v)}
decoder = {i:ch for ch, i in encoder.items()}

encode = lambda seq: [encoder[ch] for ch in seq]
decode = lambda nums: [decoder[num] for num in nums]

def prepare(data, encode):    
    d = torch.tensor(encode(data), dtype=torch.long).to(device)
    m = len(d)
    print(f'Number of T = {m:,}\n\n')
    train = d[: int(0.9*m)]
    test = d[int(0.9*m):]
    return train, test

def load_batch(data, batches=125, batch_size=8, block_size=8):
    for _ in range(0, batches):
        random_values = torch.randint(low=0, high=len(data)-block_size, size=(batch_size,))
        X = torch.stack([data[x:x+block_size] for x in random_values]).to(device)
        Y = torch.stack([data[x+1:x+1+block_size] for x in random_values]).to(device)
        yield X, Y

@torch.no_grad()
def evaluate(model, TRAIN, EVAL):
    model.eval()
    losses = []
    for X, Y in load_batch(TRAIN, batches=2_641):
        Z, loss = model(X, Y)
        losses.append(loss.item())
    o_loss = torch.tensor(losses).mean()

    losses = []
    for X, Y in load_batch(EVAL, batches=2_641):
        Z, loss = model(X, Y)
        losses.append(loss.item())
    e_loss = torch.tensor(losses).mean()

    model.train()
    return e_loss, o_loss

def checkpoint_model(model, optimizer, e_loss, epoch):
    best_loss = model.model_state['best_loss']
    if e_loss < best_loss:
        model.model_state['best_loss'] = e_loss
        
        PATH = os.path.join(DIR, f'model--{epoch}.pt')

        check_point = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': e_loss,
            'epoch': epoch,
        }
        print(f'Model checkpoint @{epoch=}')
        torch.save(check_point, f=PATH)

def from_pretrained(model, optimizer, PATH):
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        print(f'Loading pre-trained Model \t L=={checkpoint["loss"]}')

        model_state = checkpoint['model_state']
        optimizer_state = checkpoint['optimizer_state']

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

        return model, optimizer
    else:
        print(f'{PATH=} NOT FOUND')
        print('Loading random W and b')
        return model, optimizer

def eval_and_logging(n_batch, model, optimizer, TRAIN, EVAL, cut=1_000):
    if n_batch % cut == 0 or n_batch == 23_774:
        e_loss, o_loss = evaluate(model, TRAIN, EVAL)
        print(f'{n_batch=:,}\nEVAL={e_loss:.4f}\tTRAIN={o_loss:.4f}\n')
        checkpoint_model(model, optimizer, e_loss, n_batch)

def run(conf):
    TRAIN, EVAL = prepare(data, encode)
    batches = int((len(TRAIN) * 0.9) / 8)

    model = GPTModel(conf).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=0)
    print(f'Number of P={model.number_of_parameters():,}')
    model, optimizer = from_pretrained(model, optimizer, 'Models/model--21000.pt')

    L = []
    epochs = 1
    for e in range(epochs):

        for b , (X, Y) in tqdm(enumerate(load_batch(TRAIN, batches=batches))):
            eval_and_logging(b, model, optimizer, TRAIN, EVAL)
            Z, loss = model(X, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            L.append(loss.item())
    return model, L
