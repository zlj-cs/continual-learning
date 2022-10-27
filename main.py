import torch
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Tuple
import numpy as np 
import random
import argparse
import torch.nn as nn
import copy

class DebugDataset(Dataset):
    def __init__(self, mode: str) -> None:
        super().__init__()
        assert mode in ['train', 'test']
        if mode == 'train':
            self.x = torch.randn([100, 768])
            self.y = torch.randint(0, 2, size=[100, 1], dtype=torch.float32)
        else:
            self.x = torch.randn([10, 768])
            self.y = torch.randint(0, 2, size=[10, 1])

    def __getitem__(self, index):
        return self.x[index, :], self.y[index, :]
    
    def __len__(self):
        return len(self.x)

class Buffer():
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        self.n_memories = args.n_memories
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.memory_data = torch.FloatTensor(1 + n_tasks, self.n_memories, n_inputs)
        self.memory_labels = torch.FloatTensor(1 + n_tasks, self.n_memories)
        self.device = args.device
        if self.device:
            self.memory_data.to(self.device)
            self.memory_labels.to(self.device)

        self.ref_size = args.ref_size
        self.buffer_size = 0
        self.observed_tasks = set()
         
    
    def update(self, D_t: Dataset, t: int):
        """
        D_t:
            train dataset
        """
        self.observed_tasks.add(t)
        select_indexes = torch.randint(0, len(D_t), (self.n_memories, ))
        cnt = 0
        for index in select_indexes:
            x, y = D_t.__getitem__(index)
            self.memory_data[t, cnt, :].copy_(x.data)
            self.memory_labels[t, cnt: cnt+1].copy_(y.data)
            cnt += 1
    
    def sample(self):
        each_task_sample_size = self.ref_size // len(self.observed_tasks)
        begin_idx = torch.randint(0, self.n_memories - each_task_sample_size, (1,))
        sample_x = self.memory_data[: len(self.observed_tasks), begin_idx: begin_idx+each_task_sample_size, :].reshape(-1, self.n_inputs)
        sample_y = self.memory_labels[: len(self.observed_tasks), begin_idx: begin_idx+each_task_sample_size].reshape(-1, self.n_outputs)
        # print("sample_x.shape,",sample_x.shape)
        # print("sample_y.shape,",sample_y.shape)

        return sample_x, sample_y
       

def acc(pred, true):
    return np.dot(pred.flatten(), true.flatten()) / len(pred)


def overwrite_grad(pp, newgrad, grad_dims, factor):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient, with gradient
        grad_dims: list storing number of parameters at each layer
        facotr: the facotr of gradient correction
    """
    cnt = 0
    beg = 0
    for param in pp:
        if param.grad is not None:
            pp_len = grad_dims[cnt]
            en = beg + pp_len
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data = param.grad.data - factor * this_grad.data
            beg = en
        cnt += 1


def eval_task(model: torch.nn.Module, D_test: List[Tuple[torch.Tensor, torch.LongTensor]], metric: Callable=acc):
    results = []
    for d_te in D_test:
        x, y = d_te.x, d_te.y
        pred = model(x)
        pred = pred > 0.5
        results.append(metric(pred.cpu().numpy(), y.cpu().numpy()))
    return results

STRATEGY_DICT = {}

def register(target):
    def f(k, v):
        STRATEGY_DICT[k] = v
        return v 

    if callable(target):
        return f(target.__name__, target)
    else:
        return lambda x: f(target, x)
    

class IncreamentalStrategyFactory():
    def __init__(self) -> None:
        super().__init__()
    
    @register('agem')
    def agem(model: nn.Module, 
             new_d_tr: Dataset, 
             new_d_test: Dataset, 
             buffer: Buffer,
             args,
             metirc='acc'):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
            
        # train
        for i, d_tr in enumerate(new_d_tr):
            dl = torch.utils.data.DataLoader(d_tr, args.batch_size)
            model.train()
            for _ in range(args.n_epochs):
                for x, y in dl:
                    x_ref, y_ref = buffer.sample()
                    model.zero_grad()
                    # compute past gradient
                    y_ref_pred = model(x_ref)
                    ref_loss = criterion(y_ref, y_ref_pred)
                    ref_loss.backward()
                    g_ref = [param.grad.data.view(-1) for param in model.parameters() if param.grad is not None]
                    g_ref = torch.cat(g_ref, dim=0)
                    
                    # compute batch gradient for now
                    model.zero_grad()
                    y_pred = model(x)
                    loss = criterion(y, y_pred)
                    loss.backward()

                    # correct gradient
                    g = [param.grad.data.view(-1) for param in model.parameters() if param.grad is not None]
                    g = torch.cat(g, dim=0)

                    condition = torch.dot(g, g_ref)
                    if condition < 0:               
                        factor = condition / torch.dot(g_ref, g_ref)
                        overwrite_grad(model.parameters(), g_ref, grad_dims, factor)
                    # print('g_ref.shape:', g_ref.shape)

                    optimizer.step()
                result = eval_task(model, new_d_test)
                print(result)
            buffer.update(d_tr, i+1)

    @register('vanilla')
    def vanilla_experience_replay(model,
                                  new_d_tr: Dataset, 
                                  new_d_test: Dataset, 
                                  buffer: Buffer,
                                  args,
                                  **kwargs):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        # train
        for i, d_tr in enumerate(new_d_tr):
            dl = torch.utils.data.DataLoader(d_tr, args.batch_size)           
            model.train()
            for _ in range(args.n_epochs):
                for x, y in dl:
                    x_ref, y_ref = buffer.sample()
                    model.zero_grad()
                    # compute past gradient
                    y_ref_pred = model(x_ref)
                    ref_loss = criterion(y_ref, y_ref_pred)
                    
                    # compute batch gradient for now
                    y_pred = model(x)
                    cur_loss = criterion(y, y_pred)

                    # balance current loss and past loss
                    loss = cur_loss + args.lamda * ref_loss
                    loss.backward()
    
                    optimizer.step()
                result = eval_task(model, new_d_test)
                print(result)
            buffer.update(d_tr, i+1)
    
    @register('lwf')
    def lwf(model,
            new_d_tr: Dataset, 
            new_d_test: Dataset, 
            buffer: Buffer,
            args,
            **kwargs):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        # train
        for i, d_tr in enumerate(new_d_tr):
            dl = torch.utils.data.DataLoader(d_tr, args.batch_size)           
            old_model = copy.deepcopy(model)
            model.train()
            for _ in range(args.n_epochs):
                for x, y in dl:
                    # compute old prediction
                    with torch.no_grad():
                        y_pred_old = old_model(x)
                    
                    # set up new label
                    y_pred = model(x)
                    y = (1 - args.lamda) * y  + args.lamda * y_pred_old
                    loss = criterion(y, y_pred)
                    loss.backward()
    
                    optimizer.step()
                result = eval_task(model, new_d_test)
                print(result)

class BugPartUsedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList(
            [nn.Linear(256, 10),
            nn.Linear(768, 1)]
        )

    def forward(self, x):
        return self.net[1](x)

    def show_weight(self):
        print('model.weight:', list(self.parameters())[0][0][:10])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Continuum learning')
    parser.add_argument('--strategy', type=str, default='agem', choices=['agem', 'vanilla', 'lwf'])
    parser.add_argument('--n_memories', type=int, default=100,
                        help='number of memories per task')   
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--debug', type=bool, default=False,
                        help='whether use debug mode')
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='mnist_permutations.pt',
                        help='data file')
    parser.add_argument('--model_path', default='model.pkl',
                        help='path where model is located')
    parser.add_argument('--device', default=None)
    parser.add_argument('--ref_size', default=32, type=int,
                        help='how many past samples are used to correct gradient')
    parser.add_argument('--lamda', default=0.1,
                        help='bigger lamda implies bigger influence the past experience')
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)         
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.debug:
        # prepare data
        old_tr = DebugDataset(mode='train')
        old_te = DebugDataset(mode='test')
        new_d_tr = []
        new_d_te = [old_te]
        args.n_tasks = 3
        for _ in range(3):
            new_d_tr.append(DebugDataset(mode='train'))
            new_d_te.append(DebugDataset(mode='test'))
        
        # prepare buffer
        buffer = Buffer(n_inputs=768, n_outputs=1, n_tasks=args.n_tasks, args=args)
        buffer.update(old_tr, 0)

        # prepare model
        model = BugPartUsedModel()
        
        

        # train
        model.show_weight()
        train_strategy = STRATEGY_DICT[args.strategy]
        train_strategy(model, new_d_tr, new_d_te, buffer, args)
        model.show_weight()

    else:
        # TODO:
        # 1.prepare data

        # 2.load model

        # 3.train
        pass

        



