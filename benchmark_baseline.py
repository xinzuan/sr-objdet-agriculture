import subprocess
import os

datasets = os.listdir('dataset')
dataset_configs = {
    'apple_detection_drone_brazil': 256,
    'apple_detection_spain': 512,
    'apple_detection_usa': 640,
    'fruit_detection_worldwide': 800,
    'grape_detection_californianight': 416,
    'mango_detection_australia': 512,
    'wheat_head_counting': 512
}

for dataset in datasets:
    if dataset.endswith('.zip') or dataset.endswith('.gitkeep'):
        continue

    train_command = [
        'python train.py --workers 8 --device 0 --batch-size 16 --epoch 300',
        '--cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --hyp data/hyp.scratch.custom.yaml',
        f'--data ../../dataset/{dataset}/config.yaml',
        f'--img-size {dataset_configs[dataset]} {dataset_configs[dataset]}',
        f'--name {dataset}',
        '--project ../../results/baseline/yolov7/train'
    ]
    train_command = ' '.join(train_command).split(' ')
    subprocess.call(train_command, cwd='./object_detectors/yolov7')

    test_command = [
        f'python test.py --device 0 --weights ../../results/baseline/yolov7/train/{dataset}/weights/best.pt',
        f'--data ../../dataset/{dataset}/config.yaml',
        f'--img-size {dataset_configs[dataset]} --batch 16',
        f'--task test --name {dataset}',
        '--project ../../results/baseline/yolov7/test',
    ]
    test_command = ' '.join(test_command).split(' ')
    subprocess.call(test_command, cwd='./object_detectors/yolov7')
