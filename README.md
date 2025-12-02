# OnlineTry-ON
Virtual clothes try-on using OpenCV and MediaPipe pose detection


##  Features

- Detect human pose landmarks in a model image  
- Segment and refine the torso mask  
- Automatically scale and blend clothes images onto the model  
- Save output images in `results/` folder


##  Requirements

- Python 3.8+  
- Libraries:
pip install opencv-python mediapipe numpy


##  How to Use

1. Place your model image as `model.png` in the repo root  
2. Put clothes images (`.png`, `.jpg`) in `clothes/` folder  
3. Run the script:  python second.py
