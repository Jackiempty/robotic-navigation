import os
from ultralytics import YOLO

if __name__ == "__main__":
    # Get the absolute path of the directory containing train.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Pass an absolute path for data.yaml to prevent YOLO from losing track of it due to different execution paths
    data_yaml_path = os.path.join(base_dir, "data", "data.yaml")
    
    model = YOLO("yolo26n.pt")
    model.train(data=data_yaml_path, epochs=100, batch=8)