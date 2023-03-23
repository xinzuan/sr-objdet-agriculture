
DATA_PATH=../../dataset
FOLDER_NAME=ori
AGRI=("apple_detection_drone_brazil" "apple_detection_spain" "apple_detection_usa" "fruit_detection_worldwide"  "mango_detection_australia" "wheat_head_counting" "grape_detection_californianight")

MODEL_FILE=model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth


TYPE=("train" "val" "test")
#TYPE=("val")
TILE=200
SCALE=2

for agri in "${AGRI[@]}"
    do
    for type in "${TYPE[@]}"
        do
            python main_test_swinir.py --task real_sr --scale ${SCALE} --model_path ${MODEL_FILE} \
            --folder_lq ${DATA_PATH} --tile ${TILE} --downsample \
            --type ${type} --folder_name ${FOLDER_NAME} --agri ${agri}
        done
    done



DATA_PATH=../../dataset
FOLDER_NAME=ori
AGRI=("apple_detection_drone_brazil" "apple_detection_spain" "apple_detection_usa" "fruit_detection_worldwide"  "mango_detection_australia" "wheat_head_counting" "grape_detection_californianight")

MODEL_FILE=model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth


TYPE=("train" "val" "test")
#TYPE=("val")
TILE=200
SCALE=4

for agri in "${AGRI[@]}"
    do
    for type in "${TYPE[@]}"
        do
            python main_test_swinir.py --task real_sr --scale ${SCALE} --model_path ${MODEL_FILE} \
            --folder_lq ${DATA_PATH} --tile ${TILE} --downsample \
            --type ${type} --folder_name ${FOLDER_NAME} --agri ${agri}
        done
    done
