import argparse

import torch
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from Conv_G import conv_g
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric
from sklearn import manifold, datasets
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default='./save/p2_retry-5way-1kshot-15qquery-100m-1000epoch-Conv_G/max-acc.pth')
    parser.add_argument('--batch', type=int, default=40)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15) #因為一次sample的影像數量是n_shot + q_query
    parser.add_argument('--m_augment', type=int, default=100)
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

    total_realdata = torch.empty(5, 1600)
    total_fakedata = torch.empty(5, 1600)
    #total_realdata=total_realdata.cuda()
    #total_fakedata=total_fakedata.cuda()


    for i, batch in enumerate(loader, 1):
        with torch.no_grad():  
            data, _ = [_.cuda() for _ in batch] 
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]
            print(data_shot.shape)
            #print(data_query.shape)
         
            x = model(data_shot)
            realdata = x[0:5, :]
            fakedata = x[5:10, :]
            print(realdata.shape)
            print(fakedata.shape)
            realdata = realdata.cpu()
            fakedata = fakedata.cpu() 
            total_realdata = torch.cat((total_realdata, realdata), 0 )
            total_fakedata = torch.cat((total_fakedata, fakedata), 0 )

            x = None
            realdata = None
            fakedata = None

    print(total_realdata.shape)
    print(total_fakedata.shape)
    total_realdata = total_realdata[5:,:]
    total_fakedata = total_fakedata[5:25,:]
    print(total_realdata.shape)
    print(total_fakedata.shape)
    reallabel = torch.arange(5).repeat(40)
    fakelabel = torch.arange(5).repeat(4)
    print(reallabel.shape)
    print(fakelabel.shape)

    total_realdata = total_realdata.cpu()
    total_fakedata = total_fakedata.cpu()
    reallabel = reallabel.cpu().tolist()
    fakelabel = fakelabel.cpu().tolist()


    z_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(total_realdata)
    z2_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(total_fakedata)
    #Data Visualization
    z_min, z_max = z_tsne.min(0), z_tsne.max(0)
    z2_min, z2_max = z2_tsne.min(0), z2_tsne.max(0)

    z_norm = (z_tsne - z_min) / (z_max - z_min)  #Normalize
    z2_norm = (z2_tsne - z2_min) / (z2_max - z2_min)  #Normalize

    plt.figure(figsize=(8, 8))
    for i in range(z_norm.shape[0]):
        #plt.text(z_norm[i, 0], z_norm[i, 1], str(reallabel[i]), color=plt.cm.Set1(reallabel[i]), fontdict={'weight': 'bold', 'size': 9})
        plt.plot(z_norm[i, 0], z_norm[i, 1], 'x', color=plt.cm.Set1(reallabel[i]))

    for i in range(z2_norm.shape[0]):
        #plt.text(z2_norm[i, 0], z2_norm[i, 1], str(fakelabel[i]), color=plt.cm.Set1(fakelabel[i]), fontdict={'weight': 'bold', 'size': 18})   
        plt.plot(z2_norm[i, 0], z2_norm[i, 1], '^', color=plt.cm.Set1(fakelabel[i]))
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./tsne_output.png")
    plt.show()
