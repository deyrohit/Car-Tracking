import torch
import gc
import os
import logging
from ultralytics import YOLO
logging.basicConfig(
    level=logging.INFO,
    filename="logs.txt",
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def clean_memory(model):
    if hasattr(model, 'optimizer'):
        del model.optimizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Memory cleaned successfully.")
    else:
        logger.warning("Model has no optimizer attribute to clear.")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if __name__ == '__main__':
    model = YOLO("yolov9c.pt")
    logger.info("Starting YOLO training.")
    model.train(
        data=r"C:\Users\deyro\OneDrive\Desktop\Video\Train\Data\data.yaml",
        epochs=100,
        imgsz=640,
        batch=1,
        lr0=0.0009,
        lrf=0.0008,
        momentum=0.825,
        weight_decay=0.0005,
        optimizer='AdamW',
        project=r"epoches",
        save_period=1,
        patience=10,
        iou=0.9,
        save=True,
        conf=0.9,
        close_mosaic=True,
        # resume=True,
        # degrees=0,
        #augment=False,
        #plots = True,
        # fliplr=0,
        #scale=0,
        #single_cls=False,
        mosaic=0,
        #erasing=0,
        mixup=0,
        dropout =0.3,
        #cos_lr=True,
        # translate=0,
        #crop_fraction=0,
        device=0,
        # save_txt = True,
        # save_crop = True,
        save_conf = True,
    )
    logger.info("Training completed.")
    clean_memory(model)