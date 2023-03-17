import os
import cv2
from tqdm import tqdm

# parameters
scale = 2
interpolation = "bilinear"

# We only use these three interpolations
interpolations = {
                    "bilinear": cv2.INTER_LINEAR,
                    "bicubic": cv2.INTER_CUBIC,
                    "lanczos": cv2.INTER_LANCZOS4
                }

assert interpolation in interpolations
assert scale in [2, 4]

datasets = next(os.walk('.'))[1]

def upscale_images(data_path):
    """ Upscale all images inside the 'data_path' (e.g. 'path_to_dataset/train/')
    """
    img_path = os.path.join(data_path, "images")

    for image_name in os.listdir(img_path):

        img = cv2.imread(os.path.join(img_path, image_name), cv2.IMREAD_UNCHANGED)
        # print('Original Dimensions : ',img.shape)
         
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
          
        # resize image
        resized_img = cv2.resize(img, dim, interpolation = interpolations[interpolation])
        # print('Resized Dimensions : ',resized_img.shape)

        upscaled_img_path = os.path.join(data_path, "upscaled_images_"+interpolation+"_"+str(scale)+"x")
        os.makedirs(upscaled_img_path, exist_ok=True)
        cv2.imwrite(os.path.join(upscaled_img_path, image_name), resized_img)


for dataset in tqdm(datasets):
    train_path = os.path.join(dataset, 'train')
    val_path = os.path.join(dataset, 'val')
    test_path = os.path.join(dataset, 'test')

    upscale_images(train_path)
    upscale_images(val_path)
    upscale_images(test_path)

