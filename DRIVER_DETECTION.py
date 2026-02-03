import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# --- 1. MODEL ARCHITECTURE ---
def get_driver_model(num_classes):
    # Initializing MobileNetV3 Large
    model = models.mobilenet_v3_large(weights=None)
    
    # Based on your error, the output layer needs 4 classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

# --- 2. PREPROCESSING ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_live_detection(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # --- FIXED: Changed num_classes to 4 to match your .pth file ---
    num_classes = 4 
    model = get_driver_model(num_classes).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"[INFO] Successfully loaded MobileNetV3 weights (4-class version).")
    except Exception as e:
        print(f"[ERROR] Weight mismatch: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Webcam Live. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        # --- UPDATED LABEL MAP ---
        # Since your model has 4 classes, we map them here. 
        # If you aren't sure which is which, you can check your training labels.
        label_map = {
            0: "ALERT", 
            1: "DISTRACTED",
            2: "DROWSY/OTHER", 
            3: "UNKNOWN/OTHER"
        }
        
        status = label_map.get(pred.item(), "Unknown")
        
        # Color: Green for Alert (0), Red for anything else
        color = (0, 255, 0) if pred.item() == 0 else (0, 0, 255) 

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (450, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"CONF: {conf.item()*100:.1f}%", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('Driver Detection Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    print("[PROMPT] Select your MobileNetV3 .pth file...")
    model_file = filedialog.askopenfilename(
        title="Select Model Weights (.pth)", 
        filetypes=[("PyTorch Weights", "*.pth")]
    )

    if model_file:
        run_live_detection(model_file)
    else:
        print("[EXIT] No file selected.")