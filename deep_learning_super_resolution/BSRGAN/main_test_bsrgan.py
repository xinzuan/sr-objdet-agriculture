import os.path
import logging
import torch

from utils import utils_logger
from utils import utils_image as util
# from utils import utils_model
from models.network_rrdbnet import RRDBNet as net
import argparse
from tqdm import tqdm
import glob
import shutil
"""
Spyder (Python 3.6-3.7)
PyTorch 1.4.0-1.8.1
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/BSRGAN
        https://github.com/cszn/KAIR
If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)
by Kai Zhang ( March/2020 --> March/2021 --> )
This work was previously submitted to CVPR2021.

# --------------------------------------------
@inproceedings{zhang2021designing,
  title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
  author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
  booktitle={arxiv},
  year={2021}
}
# --------------------------------------------

"""


def main():

    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version

    ap = argparse.ArgumentParser()
    ap.add_argument('--output_folder', type=str, default=None, help='output folder')
    ap.add_argument( "--folder_lq", required=True,
        help="path to image folder")
    ap.add_argument("--model_path", required=True,
        help="path to pretrained model")
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--downsample', action='store_true')
    ap.add_argument('--agri', type=str, default=None, help='Name of agriculture folder')
    ap.add_argument('--folder_name', type=str, default=None, help='Name of image folder') 
    args = ap.parse_args()





    if args.scale == 2:
        model_names = ['BSRGANx2']
    else:
        model_names = ['BSRGAN']    


    img_dir = args.folder_lq

    if args.output_folder is not None:
        save_path = args.output_folder
    else:
        save_path = args.folder_lq

    save_results = True
    sf = args.scale
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    subfolders = [ f.name for f in os.scandir(img_dir) if f.is_dir() ]
    sets =['train','val','test']
    

    model_path = args.model_path  
    model_name = model_names[0]  
    logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

    # torch.cuda.set_device(0)      # set GPU ID
    logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
    torch.cuda.empty_cache()

    # --------------------------------
    # define network and load model
    # --------------------------------
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

#            model_old = torch.load(model_path)
#            state_dict = model.state_dict()
#            for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
#                state_dict[key2] = param
#            model.load_state_dict(state_dict, strict=True)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()


    for data in subfolders:
        for t in sets:     # set model path
            dataset_folder = os.path.join( os.path.join(args.folder_lq, data), t)
            ori_val = os.path.join(dataset_folder,args.folder_name)
            # if t == 'train':
            #     ori_val = img_dir+data+'/'+t+'/ori/'
            
            save_path = f'{dataset_folder}/bsrgan_x{args.scale}'
            print(save_path)
            if os.path.exists(save_path) and os.path.isdir(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path)
            name_list = sorted(glob.glob(os.path.join(ori_val, '*')))

            for name in tqdm(name_list):
                filename = os.path.join(save_path, f'{os.path.splitext(os.path.basename(name))[0]}.jpg')

                

                # --------------------------------
                # (1) img_L
                # --------------------------------
                #idx += 1
                img_name, ext = os.path.splitext(os.path.basename(name))
                #logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name+ext))


                img_L = util.imread_uint(name, n_channels=3,ds=sf,downsample=args.downsample)

                img_L = util.uint2tensor4(img_L)
                img_L = img_L.to(device)

                # --------------------------------
                # (2) inference
                # --------------------------------
                img_E = model(img_L)

                # --------------------------------
                # (3) img_E
                # --------------------------------
                img_E = util.tensor2uint(img_E)
                if save_results:
                    util.imsave(img_E, filename)
                torch.cuda.empty_cache()
                import gc
                del img_E
                del img_L
                gc.collect()

if __name__ == '__main__':

    main()
