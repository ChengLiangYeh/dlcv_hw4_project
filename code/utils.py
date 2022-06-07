import os
#import shutil
#import time
#import pprint

import torch
import sklearn


#def set_gpu(x):
#    os.environ['CUDA_VISIBLE_DEVICES'] = x
#    print('using gpu:', x)


#def ensure_path(path):
#    if os.path.exists(path):
#        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
#            shutil.rmtree(path)
#            os.makedirs(path)
#    else:
#        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1) #因為logits有加負號，因此取argmax
    #print('pred=',pred)
    #pred = torch.argmin(logits, dim=1)
    #print('label=',label)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


#def dot_metric(a, b):
#    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    #print('query point after model=',a.shape)
    #print('mean point after model=',b.shape)
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    #print('query point after model=',a.shape)
    b = b.unsqueeze(0).expand(n, m, -1)
    #print('mean point after model=',b.shape)
    #logits = -((a - b)**2).sum(dim=2)  #print(((((a-b)**2).sum(dim=2))**0.5).shape) ?? 好像寫錯了! -> 修改成有開平方根(成功!) 
    logits = -((((a-b)**2).sum(dim=2))**0.5) #Q: 要找距離最小的，為甚麼不是直接算出來後取argmin? 而是要算出距離後加負號再取argmax? 兩個答案不一樣......
    #print(logits)
    #print(logits2)
    #print('logits=',logits.shape)
    return logits
'''
class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2
'''
