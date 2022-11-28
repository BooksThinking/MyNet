import os
from argparse import ArgumentParser
from time import time

from model_train import SWNet

import torch

parser = ArgumentParser(description='SWNet')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=320)
parser.add_argument('--phase_num', type=int, default=13)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--save_interval', type=int, default=20)
parser.add_argument('--testset_name', type=str, default='Set11')
parser.add_argument('--gpu_list', type=str, default='0')

args = parser.parse_args()

start_epoch, end_epoch = args.start_epoch, args.end_epoch
learning_rate = args.learning_rate
phase_num = args.phase_num
block_size = args.block_size

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_nf = 1  # image channel number
patch_size = 128  # training patch size
patch_number = 25600   # number of training patches
batch_size = 16
N = block_size * block_size
d = patch_size // block_size
l = d * d
cs_ratio_list = [0.01, 0.04, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]    # ratios in [0, 1] are all available

train_set_path = "./data/SET11"
log_path = "../SWNet/log"
model_dir = "../SWNet/log"
train_set = torch.randint(256, (batch_size, 3, 224, 224), dtype=torch.float32)
sample_martix = torch.randint(2, (batch_size, 224, 224), dtype=torch.float32)


model = SWNet()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=start_epoch-1)

print("traing start......")
for epoch in range(end_epoch):
    start_time = time()
    loss_avg, iter_num = 0.0, 0
    for index, data in enumerate(train_set):
        x_output = model(train_set)
        loss = (x_output - data).abs().mean()
        # zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        iter_num += 1
        loss_avg += loss.item()
    scheduler.step()

    loss_avg /= iter_num
    log_data = '[%d/%d] Average loss: %.4f, time cost: %.2fs.' % (epoch, end_epoch, loss_avg, time() - start_time)

    print(log_data)
    with open(log_path, 'a') as log_file:
        log_file.write(log_data + '\n')

    if epoch % args.save_interval == 0:
        torch.save(model.state_dict(), '%s/net_params_%d.pkl' % (model_dir, epoch))  # save only the parameters
