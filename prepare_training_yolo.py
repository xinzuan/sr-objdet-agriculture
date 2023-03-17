import os
import yaml
import argparse

parser = argparse.ArgumentParser(description='Configure training for YOLOv7')
parser.add_argument('--img-dir', help='Image directory name', type=str, default='images')
parser.add_argument('--config-name', help='Output config file name', type=str, default='config.yaml')
parser.add_argument('--dataset-path', help='Path to root dataset directory', type=str, default='dataset')
args = parser.parse_args()

dataset_path = args.dataset_path
datasets = next(os.walk(dataset_path))[1]
for dataset in datasets:
    train_path = os.path.join(os.path.abspath(dataset_path), dataset, 'train', args.img_dir)
    val_path = os.path.join(os.path.abspath(dataset_path), dataset, 'val', args.img_dir)
    test_path = os.path.join(os.path.abspath(dataset_path), dataset, 'test', args.img_dir)

    with open(os.path.join(dataset_path, dataset, 'config.yaml'), 'r') as f:
        train_config = yaml.safe_load(f)
        train_config.pop('path', None)
        train_config['train'] = train_path
        train_config['val'] = val_path
        train_config['test'] = test_path

    with open(os.path.join(dataset_path, dataset, args.config_name), 'w') as fin:
        yaml.safe_dump(train_config, fin)

    try:
        os.remove(os.path.join(os.path.abspath(dataset_path), dataset, 'train', 'labels.cache'))
        os.remove(os.path.join(os.path.abspath(dataset_path), dataset, 'val', 'labels.cache'))
        os.remove(os.path.join(os.path.abspath(dataset_path), dataset, 'test', 'labels.cache'))
    except:
        continue
    
print('Preparation complete')