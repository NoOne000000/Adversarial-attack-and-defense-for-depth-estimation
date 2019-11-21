import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import modules, net, resnet
import loaddata
import util
import numpy as np
import net_mask


def main():
    Encoder = modules.E_resnet(resnet.resnet50(pretrained=True))
    N = net.model(Encoder, num_features=2048,block_channel=[256, 512, 1024, 2048])
    G_adv = net_mask.G(net_mask.drn_d_22(pretrained=True))
  
    N = torch.nn.DataParallel(N).cuda()
    G_adv = torch.nn.DataParallel(G_adv).cuda()

    N.load_state_dict(torch.load('./models/N'))
    G_adv.load_state_dict(torch.load('./models/G_adv'))

    cudnn.benchmark = True

    test_loader = loaddata.getTestingData(8)

    #test for N(x*G_adv(x*))
    test_G_adv(test_loader, N, G_adv, epsilon=0.05, iteration=10)
    test_G_adv(test_loader, N, G_adv, epsilon=0.1, iteration=10)
    test_G_adv(test_loader, N, G_adv, epsilon=0.15, iteration=10)
    test_G_adv(test_loader, N, G_adv, epsilon=0.2, iteration=10)

def test_G_adv(test_loader, N, G_adv, epsilon=0.3, iteration=10):
    totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    N.eval()
    G_adv.eval()

    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        image = torch.autograd.Variable(image).cuda()
        depth = torch.autograd.Variable(depth).cuda()

        img_clean = image.clone()

        count = 0
        if 1:
            img_adv = image.clone()
            img_adv.requires_grad = True
            img_min = float(image.min()[0].data.cpu().numpy())
            img_max = float(image.max()[0].data.cpu().numpy())

            while count < iteration:
                output = N(img_adv)
       
                loss = torch.abs(output - depth).mean()
                N.zero_grad()
                loss.backward()
           
                img_adv.grad.sign_()
                img_adv = img_adv + img_adv.grad
                img_adv = where(img_adv > image + epsilon, image + epsilon, img_adv)
                img_adv = where(img_adv < image - epsilon, image - epsilon, img_adv)
                img_adv = torch.clamp(img_adv, img_min, img_max)
                img_adv = torch.autograd.Variable(img_adv.data, requires_grad=True)
       
                count += 1
      
        mask_adv = G_adv(img_adv)
        output = N(img_adv * mask_adv)

        batchSize = depth.size(0)
        errors = util.evaluateError(output, depth)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        totalNumber = totalNumber + batchSize
        averageError = util.averageErrors(errorSum, totalNumber)

    print('rmse:', np.sqrt(averageError['MSE']))
    print(averageError)




def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond * x) + ((1 - cond) * y)




if __name__ == '__main__':
    main()
