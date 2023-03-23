
DATA_PATH=../../dataset
FOLDER_NAME=ori
MODEL_FILE=./model_zoo/BSRGAN.pth

SCALE=4


python main_test_bsrgan.py --scale ${SCALE} --model_path ${MODEL_FILE} \
--folder_lq ${DATA_PATH} \
--folder_name ${FOLDER_NAME}

