import torch
from thop import profile
import torch.nn.functional as F
import numpy as np
import os, argparse, cv2
from scipy import misc
from lib.SCPNet import Network
from utils.data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default='/media/lab509-1/data1/RJC/ASBI-main/SSCNet(1)/snapshot/SCPNet/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
#for _data_name in ['LungInfection']:
    data_path = '/media/lab509-1/data1/RJC/CODDataset/TestDataset/{}/'.format(_data_name)
    save_path = './SCPNet/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = Network()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        S_4_pred, S_3_pred, S_2_pred, S_1_pred = model(image)
        res = S_1_pred
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {} '.format(_data_name, name))
        #misc.imsave(save_path+name, res)
        cv2.imwrite(save_path+name, res*255)

		
