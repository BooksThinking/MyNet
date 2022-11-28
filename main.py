import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 16
lr_decay = 0.8
beta1 = 0.9
n_epoch = 200
decay_every = 50
ni = int(4)
ni_ = int(batch_size//4)
lr = 0.0001
block_size = 16
MR = 0.25
num_stage = 8
imagesize = block_size * block_size
size_y = math.ceil(block_size * block_size * MR)

def parse_option():
    parser = argparse.ArgumentParser('My TestNet Transformer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU", default=16)
    parser.add_argument('--n_epoch', type=int, default=200)



if __name__ == '__main__':

