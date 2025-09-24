import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from ultralytics import YOLO

# Create a new YOLO model from the StarFreq_dyhead.yaml configuration
model = YOLO("ultralytics/cfg/models/11/StarFreq_dyhead.yaml")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="datasets/gwhd_2021_yolo/data.yaml", epochs=100, batch=64, device="0,1,2,3")

# Evaluate the model's performance on the validation set
results = model.val()
