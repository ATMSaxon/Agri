import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")


# Load a pretrained YOLO model (recommended for training)
# model = YOLO("yolo11m.pt")

# model = YOLO("ultralytics/cfg/models/11/StarFreq.yaml").load("yolo11m.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="datasets/gwhd_2021_yolo/data.yaml", epochs=100, batch=64, device="0,1,2,3")

# Evaluate the model's performance on the validation set
results = model.val()
