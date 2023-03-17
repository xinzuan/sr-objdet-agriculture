# import the necessary packages
import argparse
import time
import cv2
import os
import glob

from tqdm import tqdm
import torch
from model import FSRCNN
import numpy as np
import PIL.Image as pil_image

import shutil


#loader = agml.data.AgMLDataLoader('grape_detection_californiaday')

#path ='/home/vania/.agml/datasets/grape_detection_californiaday'


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr

def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])

ap = argparse.ArgumentParser()
ap.add_argument('--output_folder', type=str, default=None, help='output folder')
ap.add_argument( "--folder_lq", required=True,
	help="path to image folder")
ap.add_argument("--model_path", required=True,
	help="path to pretrained model")
ap.add_argument('--scale', type=int, default=3)
ap.add_argument('--downsample', action='store_true')
ap.add_argument('--agri', type=str, default=None, help='Name of agriculture folder')
ap.add_argument('--folder_name', type=str, default=None, help='Name of image folder') 
args = ap.parse_args()

img_dir = args.folder_lq
#name_list = sorted(glob.glob(os.path.join(args['folder_lq'], '*')))
if args.output_folder is not None:
    save_path = args.output_folder
else:
    save_path = args.folder_lq


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = FSRCNN(scale_factor=args.scale).to(device)

state_dict = model.state_dict()
for n, p in torch.load(args.model_path, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

model.eval()

# sr = dnn_superres.DnnSuperResImpl_create()
# path = "/media/vania/Data/uav/uav/weights/FSRCNN_x2.pb"
# sr.readModel(path)
# sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# sr.setModel("fsrcnn", 2)

#print(name_list[0])
err =[]
subfolders = [ f.name for f in os.scandir(img_dir) if f.is_dir() ]
sets =['train','val','test']
for data in subfolders:
    for t in sets:
        dataset_folder = os.path.join( os.path.join(args.folder_lq, data), t)
        ori_val = os.path.join(dataset_folder,args.folder_name)
        # if t == 'train':
        #     ori_val = img_dir+data+'/'+t+'/ori/'
        
        save_path = f'{dataset_folder}/fsrcnn_x{args.scale}'
        print(save_path)
        if os.path.exists(save_path) and os.path.isdir(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        name_list = sorted(glob.glob(os.path.join(ori_val, '*')))
        for name in tqdm(name_list):
            filename = os.path.join(save_path, f'{os.path.splitext(os.path.basename(name))[0]}.jpg')

            if os.path.exists(filename):
                continue




            try:
                img =  pil_image.open(name).convert('RGB')
                width = (img.width//args.scale)*args.scale
                height = (img.height//args.scale)*args.scale
                dim = (width, height)

                hr = img.resize(dim, resample=pil_image.BICUBIC)

                if args.downsample:
                    width = hr.width//args.scale
                    height = hr.height//args.scale
                    dim = (width, height)
                    lr = hr.resize(dim, resample=pil_image.BICUBIC)
                else:
                    lr = img
                bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            # bicubic.save(filename2)

                lr,_ = preprocess(lr,device)
                _, ycbcr = preprocess(bicubic, device)

                preds = model(lr).clamp(0.0, 1.0)

                preds = preds.mul(255.0).detach().cpu().numpy().squeeze(0).squeeze(0)

                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
                output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
                output = pil_image.fromarray(output)

                output.save(filename)
            except:
                err.append(name)

print(err)


