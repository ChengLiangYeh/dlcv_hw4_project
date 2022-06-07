import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from generator import Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-1-modifyDF-5way-1kshot-15qquery-datahalu-m100-epoch100-modifyVal-modifySave-modifyG')
    #parser.add_argument('--gpu', default='0')
    parser.add_argument('--train_data_dir')
    parser.add_argument('--train_csv')
    parser.add_argument('--val_data_dir')
    parser.add_argument('--val_csv')
    parser.add_argument('--m_augment', type=int, default=100)
    args = parser.parse_args()
    pprint(vars(args))

    #set_gpu(args.gpu)
    ensure_path(args.save_path)

    train_data_dir = args.train_data_dir
    train_csv = args.train_csv
    trainset = MiniImageNet(train_data_dir, train_csv)
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=0, pin_memory=True) #pin_memory => RAM內存的Tensor轉換到GPU的顯存就更快，但是會吃RAM

    val_data_dir = args.val_data_dir
    val_csv = args.val_csv
    valset = MiniImageNet(val_data_dir, val_csv)
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=0, pin_memory=True)

    model = Convnet().cuda()
    Generator = Generator().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizerG = torch.optim.Adam(Generator.parameters(), lr=0.001)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) #每過20個epoch => lr = lr * gamma
    lr_schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
        torch.save(Generator.state_dict(), osp.join(args.save_path, name + '_Generator.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        
        model.train()
        Generator.train()
        tl = Averager() #作者自己寫的average class
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            #print(proto.shape)
            noise = torch.randn(proto.shape[0]*args.m_augment, proto.shape[1]) ####
            noise = noise.cuda()
            #print(noise.shape)
            cat_data = torch.cat((proto, noise), 0)
            #print(cat_data.shape)
            proto = Generator(cat_data)
            #print(proto.shape)
            proto = proto.reshape(args.shot+args.m_augment, args.train_way, -1).mean(dim=0) ####
            #print(proto.shape)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            #print('label=',label.shape)
            ###############################################################################euclidean
            #print(proto.shape)
            #print(model(data_query).shape)
            logits = euclidean_metric(model(data_query), proto)
            #print('logits=',logits.shape)
            #print(logits)
            ###############################################################################cosine
            '''
            cos = torch.nn.CosineSimilarity(dim=2)
            a = model(data_query)
            b = proto
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

            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            optimizerG.zero_grad()
            loss.backward()
            optimizer.step()
            optimizerG.step()
            
            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()
        lr_scheduler.step()
        lr_schedulerG.step()

        model.eval()
        Generator.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            #print(proto.shape)
            noise = torch.randn(proto.shape[0]*args.m_augment, proto.shape[1]) ####
            noise = noise.cuda()
            #print(noise.shape)
            cat_data = torch.cat((proto, noise), 0)
            #print(cat_data.shape)
            proto = Generator(cat_data)
            #print(proto.shape)
            proto = proto.reshape(args.shot+args.m_augment, args.test_way, -1).mean(dim=0) ####
            #print(proto.shape)
            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            #print(label.shape)

            logits = euclidean_metric(model(data_query), proto)
            ###############################################################################cosine
            '''
            cos = torch.nn.CosineSimilarity(dim=2)
            a = model(data_query)
            b = proto
            n = a.shape[0]
            m = b.shape[0]
            a = a.unsqueeze(1).expand(n, m, -1)
            b = b.unsqueeze(0).expand(n, m, -1)
            #print('a=',a.shape)
            #print('b=',b.shape)
            output = cos(a, b)
            #print('output=',output.shape)
            logits = output-1 ###
            #print(logits)
            '''
            ###############################################################################
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

