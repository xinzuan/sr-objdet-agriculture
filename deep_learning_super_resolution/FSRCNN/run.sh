
DATA_PATH=../../dataset
FOLDER_NAME=ori
MODEL_FILE=./fsrcnn_x2.pth

SCALE=2


python fsrcnn.py --scale ${SCALE} --model_path ${MODEL_FILE} \
--folder_lq ${DATA_PATH} --downsample \
--folder_name ${FOLDER_NAME}


DATA_PATH=../../dataset
FOLDER_NAME=ori
MODEL_FILE=./fsrcnn_x2.pth

SCALE=2


python fsrcnn.py --scale ${SCALE} --model_path ${MODEL_FILE} \
--folder_lq ${DATA_PATH} \
--folder_name ${FOLDER_NAME}

DATA_PATH=../../dataset
FOLDER_NAME=ori
MODEL_FILE=./fsrcnn_x4.pth

SCALE=4


python fsrcnn.py --scale ${SCALE} --model_path ${MODEL_FILE} \
--folder_lq ${DATA_PATH} --downsample \
--folder_name ${FOLDER_NAME}

