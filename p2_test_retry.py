import argparse

import torch
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from Conv_G import conv_g
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default='./save/p2_retry-5way-1kshot-15qquery-10m-1000epoch-Conv_G/max-acc.pth')
    parser.add_argument('--batch', type=int, default=600)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15) #因為一次sample的影像數量是n_shot + q_query
    parser.add_argument('--m_augment', type=int, default=10)
    parser.add_argument('--test_data_dir')
    parser.add_argument('--test_csv')
    args = parser.parse_args()
    pprint(vars(args))

    test_data_dir = args.test_data_dir
    test_csv= args.test_csv

    dataset = MiniImageNet(test_data_dir, test_csv)
    sampler = CategoriesSampler(dataset.label, args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)

    model = conv_g().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()

    for i, batch in enumerate(loader, 1):  
        data, _ = [_.cuda() for _ in batch] 
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]
        print(data_shot.shape)
        print(data_query.shape)
         
        x = model(data_shot)
        x = x.reshape(args.shot+args.m_augment, args.way, -1).mean(dim=0)
        p = x
        ###############################################################################EU
        x = model.encoder(data_query)
        x = x.view(x.size(0), -1)
        logits = euclidean_metric(x, p)
        ###############################################################################cosine
        '''
        cos = torch.nn.CosineSimilarity(dim=2)
        a = model(data_query)
        b = p
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        #print('a=',a.shape)
        #print('b=',b.shape)
        output = cos(a, b)
        #print('output=',output.shape)
        #print('before',output)
        logits = output-1 ###
        #print('after',logits)
        '''
        ###############################################################################


        label = torch.arange(args.way).repeat(args.query)
        #print('label=',label)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
        x = None; p = None; logits = None