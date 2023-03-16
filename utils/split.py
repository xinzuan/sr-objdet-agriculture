import os
from random import shuffle
import shutil
import math
import yaml

datasets = next(os.walk('.'))[1]

for dataset in datasets:
    print(dataset)
    test_img_path = os.path.join(dataset, 'test', 'images')
    test_lbl_path = os.path.join(dataset, 'test', 'labels')
    train_img_path = os.path.join(dataset, 'train', 'images')
    train_lbl_path = os.path.join(dataset, 'train', 'labels')
    val_img_path = os.path.join(dataset, 'val', 'images')
    val_lbl_path = os.path.join(dataset, 'val', 'labels')

    temp_img_path = os.path.join(dataset, 'temp', 'images')
    temp_lbl_path = os.path.join(dataset, 'temp', 'labels')

    os.makedirs(temp_img_path, exist_ok=True)
    os.makedirs(temp_lbl_path, exist_ok=True)

    for image in os.listdir(test_img_path):
        shutil.copy2(os.path.join(test_img_path, image), os.path.join(temp_img_path, image))

    for image in os.listdir(train_img_path):
        shutil.copy2(os.path.join(train_img_path, image), os.path.join(temp_img_path, image))
            
    for lbl in os.listdir(test_lbl_path):
        shutil.copy2(os.path.join(test_lbl_path, lbl), os.path.join(temp_lbl_path, lbl))
        
    for lbl in os.listdir(train_lbl_path):
        shutil.copy2(os.path.join(train_lbl_path, lbl), os.path.join(temp_lbl_path, lbl))

    shutil.rmtree(test_img_path)
    shutil.rmtree(test_lbl_path)
    shutil.rmtree(train_img_path)
    shutil.rmtree(train_lbl_path)

    os.makedirs(test_img_path, exist_ok=True)
    os.makedirs(test_lbl_path, exist_ok=True)
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(train_lbl_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(val_lbl_path, exist_ok=True)

    images = os.listdir(temp_img_path)
    idx_train = math.floor(len(images) * 0.7)
    idx_val = math.floor(len(images) * 0.85)

    shuffle(images)

    for idx, image in enumerate(images):
        name, ext = os.path.splitext(image)

        if idx <= idx_train:
            shutil.copy2(os.path.join(temp_img_path, image), os.path.join(train_img_path, image))
            shutil.copy2(os.path.join(temp_lbl_path, f'{name}.txt'), os.path.join(train_lbl_path, f'{name}.txt'))
        elif idx <= idx_val:
            shutil.copy2(os.path.join(temp_img_path, image), os.path.join(val_img_path, image))
            shutil.copy2(os.path.join(temp_lbl_path, f'{name}.txt'), os.path.join(val_lbl_path, f'{name}.txt'))
        else:
            shutil.copy2(os.path.join(temp_img_path, image), os.path.join(test_img_path, image))
            shutil.copy2(os.path.join(temp_lbl_path, f'{name}.txt'), os.path.join(test_lbl_path, f'{name}.txt'))

    shutil.rmtree(os.path.join(dataset, 'temp'))

    with open(os.path.join(dataset, 'data.yaml'), 'r') as f:
        train_config = yaml.safe_load(f)
        train_config['path'] = f'/home/steve/Downloads/AGML-New/{dataset}'
        train_config['train'] = 'train/images'
        train_config['val'] = 'val/images'
        train_config['test'] = 'test/images'

        with open(os.path.join(dataset, 'train-config.yaml'), 'w') as fin:
            yaml.safe_dump(train_config, fin)



# path: /home/steve/Downloads/AGML/mango_detection_australia.v1i.yolov5pytorch  # dataset root dir
# train: train/images  # train images (relative to "path")
# val: test/images  # val images (relative to "path")
# test: test/images # test images (optional)

# nc: 1
# names: ['mango']