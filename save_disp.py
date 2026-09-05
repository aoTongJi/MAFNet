import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.BANet import BANet
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(BANet(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)

            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2)
            disp = padder.unpad(disp)
            file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            disp = disp.cpu().numpy().squeeze()
            if args.save_png:
                disp_16 = np.round(disp * 256).astype(np.uint16)
                skimage.io.imsave(file_stem, disp_16)

            # if args.save_numpy:
            #     np.save(file_stem.replace('.png', '.npy'), disp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='/home/xgw/BANet/ba_2d_final/checkpoints/kitti_v2/50000_banet.pth')
    parser.add_argument('--save_png', action='store_true', default=True, help='save output as gray images')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/StereoDatasets/kitti/2015/testing/image_2/*_10.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/StereoDatasets/kitti/2015/testing/image_3/*_10.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/StereoDatasets/kitti/2012/testing/colored_0/*_10.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/StereoDatasets/kitti/2012/testing/colored_1/*_10.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./output/kt15/disp_0")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp")
    
    args = parser.parse_args()

    demo(args)
