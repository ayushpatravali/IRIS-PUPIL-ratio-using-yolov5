# Iris & Pupil Detection using YOLOv5 and OpenCV

## Overview  
This project focuses on detecting the iris and pupil using YOLOv5 and OpenCV. The model is trained on the **irispupille dataset** and uses image enhancement techniques like CLAHE and specular reflection removal for better accuracy.  

---

## Installation & Setup  

### 1. Clone YOLOv5 Repository  
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

### 2. Download Iris Pupil Dataset  
Download the dataset from Roboflow:  
[**Iris Pupil Dataset (irispupille)**](https://universe.roboflow.com/iris-annotation/irispupille)  

Extract and place the dataset inside the `yolov5` directory.  

### 3. Train YOLOv5s Model  
Run the following command to train the model:  
```bash
python train.py --img 640 --batch 16 --epochs 50 --data iris.yaml --weights yolov5s.pt
```

After training, the best and last model weights will be saved in:  
```bash
runs/train/exp/weights/best.pt
runs/train/exp/weights/last.pt
```

---

## Running the Eye Detection Code  

### 1. Run the Jupyter Notebook
Use the provided `.ipynb` file to execute the detection pipeline. It includes:  
- Face & Eye detection using **Haar Cascade Classifier**  
- Opening the webcam and selecting the eye pair manually using keyboard input  
- Enhancing the selected eye using **CLAHE** and removing specular reflections  
- Using the **trained YOLOv5 model (best.pt)** for **iris and pupil detection**  

---

## Suggested Approach  

### Step 1: Capture  
- Detect the **face and eyes** using Haar Cascade Classifier.  
- Draw bounding boxes around the detected regions.  

### Step 2: Extract Eye  
- Manually select the eye pair using keyboard input.  
- Crop and extract the **left and right eye regions**.  

### Step 3: Image Enhancement  
- Apply **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for better visibility.  
- Remove **specular reflections** to improve feature extraction.  

### Step 4: Iris & Pupil Detection  
- Load the **trained YOLOv5 model (best.pt)**.  
- Detect and localize the **iris and pupil** in the enhanced eye images.  
- Display results with bounding boxes and calculate key ratios.  

---

## Methodology  

1. **Face & Eye Detection**  
   - Use OpenCV's Haar Cascade Classifier to detect the face and eyes.  
   - Manually select the eye pair for further processing.  

2. **Image Preprocessing**  
   - Convert the eye images to grayscale.  
   - Apply **CLAHE** to enhance details.  
   - Remove **specular reflections** using thresholding techniques.  

3. **YOLOv5-based Iris & Pupil Detection**  
   - Train YOLOv5 on the **irispupille dataset**.  
   - Use the trained **best.pt** model to detect the **iris and pupil**.  

4. **Result & Evaluation**  
   - Display bounding boxes for iris and pupil.  
   - Compute **iris-to-pupil ratio** for validation.  

---

## Dependencies  
Install the required libraries before running the code:  
```bash
pip install opencv-python numpy torch torchvision torchaudio matplotlib
```

---

## Conclusion  
This project provides an efficient pipeline for **real-time iris and pupil detection** using **YOLOv5 and OpenCV** with image enhancement techniques. The trained model offers **accurate eye tracking** and can be further optimized for biometric and medical applications.  
