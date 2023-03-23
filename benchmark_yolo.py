import os
import yaml
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Configure benchmark for YOLOv7')
parser.add_argument('--img-dir', help='Image directory name', type=str, required=True)
parser.add_argument('--benchmark-type', help='Benchmark name, i.e. baseline, fsrcnn_x2, etc.', type=str, required=True)
parser.add_argument('--sr-multiplier', type=int, help='Resolution upscaling multiplier, i.e. 1 for baseline', required=True)
parser.add_argument('--device', default='0', help='Cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--epoch', type=int, default=300, help='Number of epochs')
parser.add_argument('--dataset-path', type=str, default='dataset', help='Path to root dataset directory')
args = parser.parse_args()

dataset_configs = {
    'apple_detection_drone_brazil': {
        'img_res': 256,
        'total_train_imgs': 483
    },
    'apple_detection_spain': {
        'img_res': 512,
        'total_train_imgs': 677
    },
    'apple_detection_usa': {
        'img_res': 640,
        'total_train_imgs': 1598
    },
    'fruit_detection_worldwide': {
        'img_res': 800,
        'total_train_imgs': 395
    },
    'grape_detection_californianight': {
        'img_res': 416,
        'total_train_imgs': 106
    },
    'mango_detection_australia': {
        'img_res': 512,
        'total_train_imgs': 870
    },
    'wheat_head_counting': {
        'img_res': 512,
        'total_train_imgs': 4558
    },
}

def run_if_exists(path, func):
    if os.path.exists(path):
        func(path)

dataset_path = args.dataset_path
datasets = next(os.walk(dataset_path))[1]
for dataset in datasets:
    logger.info(f'Preparing {dataset}')
    d_train_path = os.path.join(os.path.abspath(dataset_path), dataset, 'train', args.img_dir)
    d_val_path = os.path.join(os.path.abspath(dataset_path), dataset, 'val', args.img_dir)
    d_test_path = os.path.join(os.path.abspath(dataset_path), dataset, 'test', args.img_dir)

    train_path = os.path.join(os.path.abspath(dataset_path), dataset, 'train', 'images')
    val_path = os.path.join(os.path.abspath(dataset_path), dataset, 'val', 'images')
    test_path = os.path.join(os.path.abspath(dataset_path), dataset, 'test', 'images')

    assert os.path.exists(d_train_path), f'Train path {d_train_path} does not exist'
    assert os.path.exists(d_val_path), f'Val path {d_val_path} does not exist'
    assert os.path.exists(d_test_path), f'Test path {d_test_path} does not exist'

    run_if_exists(os.path.join(dataset_path, dataset, 'config.yaml'), os.remove)
    run_if_exists(train_path, os.unlink)
    run_if_exists(val_path, os.unlink)
    run_if_exists(test_path, os.unlink)
    os.symlink(d_train_path, train_path)
    os.symlink(d_val_path, val_path)
    os.symlink(d_test_path, test_path)

    with open(os.path.join(dataset_path, dataset, 'base.yaml'), 'r') as f:
        train_config = yaml.safe_load(f)
        train_config.pop('path', None)
        train_config['train'] = train_path
        train_config['val'] = val_path
        train_config['test'] = test_path

    print(train_config)
    with open(os.path.join(dataset_path, dataset, 'config.yaml'), 'w') as fin:
        yaml.safe_dump(train_config, fin)

    run_if_exists(os.path.join(os.path.abspath(dataset_path), dataset, 'train', 'labels.cache'), os.remove)
    run_if_exists(os.path.join(os.path.abspath(dataset_path), dataset, 'val', 'labels.cache'), os.remove)
    run_if_exists(os.path.join(os.path.abspath(dataset_path), dataset, 'test', 'labels.cache'), os.remove)

    total_train_imgs = dataset_configs[dataset]['total_train_imgs']
    img_res = dataset_configs[dataset]['img_res']

    train_command = [
        f'python train.py --workers {args.workers} --device {args.device} --batch-size {args.batch_size} --epoch {args.epoch}',
        '--cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --hyp data/hyp.scratch.custom.yaml',
        f'--data ../../dataset/{dataset}/config.yaml',
        f'--img-size {img_res * args.sr_multiplier} {img_res * args.sr_multiplier}',
        f'--name {dataset}',
        f'--project ../../results/{args.benchmark_type}/yolov7/train'
    ]
    train_command = ' '.join(train_command)
    logger.info(f'Running training command: {train_command}')
    subprocess.call(train_command.split(' '), cwd='./object_detectors/yolov7')

    test_command = [
        f'python test.py --device 0 --weights ../../results/{args.benchmark_type}/yolov7/train/{dataset}/weights/best.pt',
        f'--data ../../dataset/{dataset}/config.yaml',
        f'--img-size {img_res * args.sr_multiplier} --batch 16',
        f'--task test --name {dataset}',
        f'--project ../../results/{args.benchmark_type}/yolov7/test',
    ]
    test_command = ' '.join(test_command)
    logger.info(f'Running test command: {test_command}')
    subprocess.call(test_command.split(' '), cwd='./object_detectors/yolov7')

    logger.info(f'Cleaning up {dataset}')
    run_if_exists(os.path.join(dataset_path, dataset, 'config.yaml'), os.remove)
    run_if_exists(train_path, os.unlink)
    run_if_exists(val_path, os.unlink)
    run_if_exists(test_path, os.unlink)
    logger.info(f'Finished {dataset}\n')
    
logger.info(f'Benchmark {args.benchmark_type} completed')