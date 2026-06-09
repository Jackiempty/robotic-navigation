import os
from ultralytics import YOLO

def analyze_segmentation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    seg_model_path = os.path.join(base_dir, "seg_best.pt")
    images_dir = os.path.join(base_dir, "segmentation", "data", "valid", "images")
    
    if not os.path.exists(images_dir) or not os.path.exists(seg_model_path):
        print("Paths not found. Please ensure seg_best.pt and validation images exist.")
        return

    print(f"Loading Segmentation Model: {seg_model_path}")
    model = YOLO(seg_model_path)
    
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    print(f"\nEvaluating on {len(images)} validation images to analyze 'road' vs 'bridge' detections...\n")
    
    road_confidences = []
    bridge_confidences = []
    bridge_detected_images = 0
    road_detected_images = 0

    # Run inference on all validation images
    results = model.predict(source=images, stream=True, verbose=False)
    
    for r in results:
        names = r.names
        has_bridge = False
        has_road = False
        
        if r.boxes is not None and r.boxes.cls is not None:
            classes = r.boxes.cls.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            
            for cls_idx, conf in zip(classes, confs):
                class_name = names[int(cls_idx)].lower()
                if 'road' in class_name:
                    road_confidences.append(conf)
                    has_road = True
                elif 'bridge' in class_name:
                    bridge_confidences.append(conf)
                    has_bridge = True
                    
        if has_bridge: bridge_detected_images += 1
        if has_road: road_detected_images += 1

    avg_road_conf = sum(road_confidences) / len(road_confidences) if road_confidences else 0
    avg_bridge_conf = sum(bridge_confidences) / len(bridge_confidences) if bridge_confidences else 0

    print("="*40)
    print("         SEGMENTATION ANALYSIS")
    print("="*40)
    print(f"Total Valid Images Analyzed: {len(images)}")
    print(f"Images containing 'road' predictions:   {road_detected_images} ({(road_detected_images/len(images))*100:.1f}%)")
    print(f"Images containing 'bridge' predictions: {bridge_detected_images} ({(bridge_detected_images/len(images))*100:.1f}%)")
    print("-" * 40)
    print(f"Total 'road' instances detected:   {len(road_confidences)}")
    print(f"Total 'bridge' instances detected: {len(bridge_confidences)}")
    print("-" * 40)
    print(f"Average Confidence ('road'):   {avg_road_conf:.4f}")
    print(f"Average Confidence ('bridge'): {avg_bridge_conf:.4f}")
    print("="*40)

if __name__ == "__main__":
    analyze_segmentation()
