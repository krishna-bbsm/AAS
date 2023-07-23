# Automated Attendance System
# Overview
The Automated Attendance System is an innovative application that leverages the power of machine learning and computer vision to automate the process of attendance marking. It utilizes a pre-trained model for face recognition and OpenCV for face detection, offering the capability to detect and recognize multiple faces in an image. The recognized faces are then cross-referenced with an existing database, and the attendance is marked with a timestamp in an Excel sheet.

# Key Features
Face Detection: Utilizes OpenCV's Haar Cascade classifier for efficient and accurate face detection.
Face Recognition: Employs a pre-trained model (VGGFace) for robust face recognition.
Automated Attendance: Marks attendance with a timestamp in an Excel sheet automatically.
Multi-Face Handling: Capable of detecting and recognizing multiple faces in a single image.
Versatile Recognition: Can recognize both frontal and profile faces.
Usage
Environment Setup: Run the code in a Python environment. The code is compatible with Google Colab.
Image Collection: The system will automatically mount your Google Drive and set up the working directory. It will then collect images from the specified directory and extract faces from them.
Face Processing: The extracted faces are processed and their representations are computed using the VGGFace model.
Image Capture: The system will capture an image using the webcam. The captured image is processed, and faces in the image are detected.
Face Recognition: The detected faces are recognized by comparing their representations with the ones computed earlier.
Attendance Marking: If a match is found, the system will mark the attendance of the recognized person in an Excel sheet with the current timestamp.

# Dependencies
OpenCV
TensorFlow
Keras
openpyxl
PIL
ipywidgets
numpy

# Important Notes
Ensure to set the correct paths for the image directory, Haar Cascade files, and the Excel sheet.
The system requires a pre-trained model for face recognition. The VGGFace model can be downloaded from Kaggle.
The system uses the cosine similarity to match faces. Adjust the similarity threshold according to your needs for optimal results.
