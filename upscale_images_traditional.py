import os
import cv2
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Upscale images with traditional approaches')
parser.add_argument('--scale', type=int, help='scaling factor', choices=[2, 4], default=2)
parser.add_argument('--interpolation', help='interpolation method', choices=['bilinear', 'bicubic', 'lanczos'], default='bilinear')
parser.add_argument('--dataset-path', help='path to the list of dataset directory', default='dataset/')
parser.add_argument('--img-dir', help='directory of images', default='images')
args = parser.parse_args()

interpolations = {
                    "bilinear": cv2.INTER_LINEAR,
                    "bicubic": cv2.INTER_CUBIC,
                    "lanczos": cv2.INTER_LANCZOS4
                }

datasets = next(os.walk(args.dataset_path))[1]

def upscale_images(data_path):
    """ Upscale all images inside the 'data_path' (e.g. 'path_to_dataset/train/')
    """
    img_path = os.path.join(data_path, args.img_dir)

    for image_name in tqdm(os.listdir(img_path)):

        img = cv2.imread(os.path.join(img_path, image_name), cv2.IMREAD_UNCHANGED)
        # print('Original Dimensions : ',img.shape)
         
        width = int(img.shape[1] * args.scale)
        height = int(img.shape[0] * args.scale)
        dim = (width, height)
          
        # resize image
        resized_img = cv2.resize(img, dim, interpolation = interpolations[args.interpolation])
        # print('Resized Dimensions : ',resized_img.shape)

        upscaled_img_path = os.path.join(data_path, "upscaled_images_"+args.interpolation+"_x"+str(args.scale))
        os.makedirs(upscaled_img_path, exist_ok=True)
        cv2.imwrite(os.path.join(upscaled_img_path, image_name), resized_img)


for dataset in datasets:
    print(f'Upscaling images in dataset: {dataset}')
    train_path = os.path.join(args.dataset_path, dataset, 'train')
    val_path = os.path.join(args.dataset_path, dataset, 'val')
    test_path = os.path.join(args.dataset_path, dataset, 'test')

    upscale_images(train_path)
    upscale_images(val_path)
    upscale_images(test_path)

