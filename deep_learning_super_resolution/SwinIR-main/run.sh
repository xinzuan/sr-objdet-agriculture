
DATA_PATH=../../dataset
FOLDER_NAME=ori
#AGRI=("apple_detection_spain.v1i.yolov5pytorch" "apple_detection_usa.v1i.yolov5pytorch" "fruit_detection_worldwide.v1i.yolov5pytorch"  "mango_detection_australia.v1i.yolov5pytorch" )
#AGRI=("grape_detection_californiaday.v2i.yolov5pytorch" "grape_detection_californianight.v1i.yolov5pytorch" "grape_detection_syntheticday.v1i.yolov5pytorch" "mango_detection_australia.v1i.yolov5pytorch" "plant_doc_detection.v1i.yolov5pytorch")
AGRI=("grape_detection_californianight.v1i.yolov5pytorch")
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
            --folder_lq ${DATA_PATH} --tile ${TILE} \
            --type ${type} --folder_name ${FOLDER_NAME} --agri ${agri}
        done
    done

