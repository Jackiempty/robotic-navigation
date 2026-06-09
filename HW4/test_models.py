import os
import random
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Model paths
    det_model_path = os.path.join(base_dir, "det_best.pt")
    seg_model_path = os.path.join(base_dir, "seg_best.pt")
    
    # Test images directory
    images_dir = os.path.join(base_dir, "segmentation", "data", "valid", "images")
    
    # Pick a sample image (first one in the directory)
    if os.path.exists(images_dir):
        images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        if images:
            test_image = os.path.join(images_dir, random.choice(images))
            print(f"Testing on random image: {test_image}")
        else:
            print(f"No images found in {images_dir}.")
            return
    else:
        print(f"Directory {images_dir} does not exist.")
        return

    # Load models
    print(f"Loading Detection Model: {det_model_path}")
    model_det = YOLO(det_model_path)
    
    print(f"Loading Segmentation Model: {seg_model_path}")
    model_seg = YOLO(seg_model_path)

    # Inference Path
    output_dir = os.path.join(base_dir, "inference_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Running Inference for Detection...")
    results_det = model_det.predict(
        source=test_image,
        save=True,
        project=output_dir,
        name="detection",
        exist_ok=True
    )
    
    print("Running Inference for Segmentation...")
    results_seg = model_seg.predict(
        source=test_image,
        save=True,
        project=output_dir,
        name="segmentation",
        exist_ok=True
    )
    
    print("\nInference Complete!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
