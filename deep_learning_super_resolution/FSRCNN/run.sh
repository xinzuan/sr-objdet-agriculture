
DATA_PATH=../../dataset
FOLDER_NAME=ori
MODEL_FILE=./fsrcnn_x2.pth

SCALE=2


python fsrcnn.py --scale ${SCALE} --model_path ${MODEL_FILE} \
--folder_lq ${DATA_PATH} \
--folder_name ${FOLDER_NAME}

