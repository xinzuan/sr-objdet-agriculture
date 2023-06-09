import os
import yaml
import argparse
import subprocess
import logging
import json
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Configure benchmark for Faster R-CNN')
parser.add_argument('--img-dir', help='Image directory name', type=str, required=True)
# parser.add_argument('--benchmark-type', help='Benchmark name, i.e. baseline, fsrcnn_x2, etc.', type=str, required=True)
# parser.add_argument('--sr-multiplier', type=int, help='Resolution upscaling multiplier, i.e. 1 for baseline', required=True)
# parser.add_argument('--device', default='0', help='Cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
# parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
# parser.add_argument('--total-iterations', type=int, default=5000, help='Total training iterations')
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

def create_coco_annotations(img_lbl_path, img_dir, config_yaml):
    
    img_path = os.path.join(img_lbl_path, img_dir)
    lbl_path = os.path.join(img_lbl_path, 'labels')

    result = {
        'info': {
            'year': '2023',
            'description': 'Autogenerated COCO annotations',
        },
        'licenses': [{
            'id': 1,
            'url': 'https://creativecommons.org/licenses/by/4.0/',
            'name': 'CC BY 4.0'
        }],
        'categories': [
            {
                'id': i + 1,
                'name': config_yaml['names'][i],
                'supercategory': config_yaml['names'][i],
            }
            for i in range(len(config_yaml['names']))
        ],
        'images': [],
        'annotations': []
    }

    assert os.path.exists(img_path), f'Image path {img_path} does not exist'
    assert os.path.exists(lbl_path), f'Label path {lbl_path} does not exist'

    img_id = 0
    annotation_id = 0
    img_files = next(os.walk(img_path))[2]
    for img in tqdm(img_files):
        img_name = os.path.splitext(img)[0]
        label_file = os.path.join(lbl_path, f'{img_name}.txt')

        assert os.path.exists(label_file), f'Label file {label_file} does not exist'

        img_file = Image.open(os.path.join(img_path, img))
        img_width, img_height = img_file.size
        
        with open(label_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = line.split(' ')
                category_id = int(line[0]) + 1
                x = int(float(line[1]) * img_width)
                y = int(float(line[2]) * img_height)
                w = int(float(line[3]) * img_width)
                h = int(float(line[4]) * img_height)

                result['annotations'].append({
                    'id': annotation_id,
                    'image_id': img_id,
                    'category_id': category_id,
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'segmentation': [],
                    'iscrowd': 0,
                })
                annotation_id += 1

        result['images'].append({
            'id': img_id,
            'license': 1,
            'file_name': img,
            'height': img_height,
            'width': img_width,
        })
        img_id += 1
    json.dump(result, open(os.path.join(img_lbl_path, 'annotations.json'), 'w'))

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

    run_if_exists(train_path, os.unlink)
    run_if_exists(val_path, os.unlink)
    run_if_exists(test_path, os.unlink)
    os.symlink(d_train_path, train_path)
    os.symlink(d_val_path, val_path)
    os.symlink(d_test_path, test_path)

    with open(os.path.join(dataset_path, dataset, 'base.yaml'), 'r') as f:
        train_config = yaml.safe_load(f)

    base_dataset_path = os.path.join(os.path.abspath(dataset_path), dataset)
    run_if_exists(os.path.join(base_dataset_path, 'train', 'annotations.json'), os.remove)
    run_if_exists(os.path.join(base_dataset_path, 'val', 'annotations.json'), os.remove)
    run_if_exists(os.path.join(base_dataset_path, 'test', 'annotations.json'), os.remove)
    logger.info(f'Creating COCO annotations for {dataset}')

    create_coco_annotations(os.path.join(base_dataset_path, 'train'), args.img_dir, train_config)
    create_coco_annotations(os.path.join(base_dataset_path, 'val'), args.img_dir, train_config)
    create_coco_annotations(os.path.join(base_dataset_path, 'test'), args.img_dir, train_config)
    
    # total_train_imgs = dataset_configs[dataset]['total_train_imgs']
    # img_res = dataset_configs[dataset]['img_res']
    # epoch = args.total_iterations // (total_train_imgs // args.batch_size)

    # train_command = [
    #     f'python train.py --workers {args.workers} --device {args.device} --batch-size {args.batch_size} --epoch {epoch}',
    #     '--cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --hyp data/hyp.scratch.custom.yaml',
    #     f'--data ../../dataset/{dataset}/config.yaml',
    #     f'--img-size {img_res * args.sr_multiplier} {img_res * args.sr_multiplier}',
    #     f'--name {dataset}',
    #     f'--project ../../results/{args.benchmark_type}/yolov7/train'
    # ]
    # train_command = ' '.join(train_command)
    # logger.info(f'Running training command: {train_command}')
    # subprocess.call(train_command.split(' '), cwd='./object_detectors/yolov7')

    # test_command = [
    #     f'python test.py --device 0 --weights ../../results/{args.benchmark_type}/yolov7/train/{dataset}/weights/best.pt',
    #     f'--data ../../dataset/{dataset}/config.yaml',
    #     f'--img-size {img_res * args.sr_multiplier} --batch 16',
    #     f'--task test --name {dataset}',
    #     f'--project ../../results/{args.benchmark_type}/yolov7/test',
    # ]
    # test_command = ' '.join(test_command)
    # logger.info(f'Running test command: {test_command}')
    # subprocess.call(test_command.split(' '), cwd='./object_detectors/yolov7')

    logger.info(f'Cleaning up {dataset}')
    run_if_exists(os.path.join(base_dataset_path, 'train', 'annotations.json'), os.remove)
    run_if_exists(os.path.join(base_dataset_path, 'val', 'annotations.json'), os.remove)
    run_if_exists(os.path.join(base_dataset_path, 'test', 'annotations.json'), os.remove)
    run_if_exists(train_path, os.unlink)
    run_if_exists(val_path, os.unlink)
    run_if_exists(test_path, os.unlink)
    logger.info(f'Finished {dataset}\n')
    
# logger.info(f'Benchmark {args.benchmark_type} completed')