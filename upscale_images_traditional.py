import os
import cv2
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Upscale images with traditional approaches')
parser.add_argument('--scale', type=int, help='scaling factor', choices=[2, 4], default=2)
parser.add_argument('--interpolation', help='interpolation method', choices=['bilinear', 'bicubic', 'lanczos'], default='bilinear')
parser.add_argument('--dataset-path', help='path to the list of dataset directory', default='dataset/')
parser.add_argument('--img-dir', help='directory of images', default='images')
parser.add_argument('--downscale-first', help='downscale first before upscale', action='store_true')
args = parser.parse_args()

interpolations = {
                    "bilinear": cv2.INTER_LINEAR,
                    "bicubic": cv2.INTER_CUBIC,
                    "lanczos": cv2.INTER_LANCZOS4
                }

def upscale_images(data_path):
    """ Upscale all images inside the 'data_path' (e.g. 'path_to_dataset/train/')
    """
    img_path = os.path.join(data_path, args.img_dir)

    # create the output folder
    upscaled_img_dir = "upscaled_images_"+args.interpolation+"_x"+str(args.scale)
    if args.downscale_first:
        upscaled_img_dir = "downscaled_" + upscaled_img_dir
    upscaled_img_path = os.path.join(data_path, upscaled_img_dir)
    os.makedirs(upscaled_img_path, exist_ok=True)
    
    for image_name in tqdm(os.listdir(img_path)):

        # read the image
        img = cv2.imread(os.path.join(img_path, image_name), cv2.IMREAD_UNCHANGED)
        
        if args.downscale_first: # downscaling the image
            width = int(img.shape[1] * (1/args.scale))
            height = int(img.shape[0] * (1/args.scale))
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = interpolations[args.interpolation])

        # upscaling the image
        width = int(img.shape[1] * args.scale)
        height = int(img.shape[0] * args.scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = interpolations[args.interpolation])

        # save the image 
        cv2.imwrite(os.path.join(upscaled_img_path, image_name), img)


datasets = next(os.walk(args.dataset_path))[1]

for dataset in datasets:
    print(f'Upscaling images in dataset: {dataset}')
    train_path = os.path.join(args.dataset_path, dataset, 'train')
    val_path = os.path.join(args.dataset_path, dataset, 'val')
    test_path = os.path.join(args.dataset_path, dataset, 'test')

    upscale_images(train_path)
    upscale_images(val_path)
    upscale_images(test_path)

