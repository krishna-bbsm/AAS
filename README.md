Automated Attendance System
Project Description
The Automated Attendance System is a machine learning project that uses face recognition to automate the process of attendance marking. The system uses a pre-trained model for face recognition, and OpenCV for face detection. It is capable of detecting and recognizing multiple faces in an image. The recognized faces are then matched with the existing database, and the attendance is marked with the timestamp in an Excel sheet.

Features
Face detection using OpenCV's Haar Cascade classifier.
Face recognition using a pre-trained model (VGGFace).
Attendance marking with timestamp in an Excel sheet.
The system can handle multiple faces in a single image.
The system can recognize both frontal and profile faces.
How to Use
Run the code in a Python environment. The code is compatible with Google Colab.
The system will automatically mount your Google Drive and set up the working directory.
The system will then collect images from the specified directory and extract faces from them.
The extracted faces are then processed and their representations are computed using the VGGFace model.
The system will then capture an image using the webcam. The captured image is processed, and faces in the image are detected.
The detected faces are then recognized by comparing their representations with the ones computed earlier.
If a match is found, the system will mark the attendance of the recognized person in an Excel sheet with the current timestamp.
Dependencies
OpenCV
TensorFlow
Keras
openpyxl
PIL
ipywidgets
numpy
Installation
The required libraries can be installed using pip:

bash
Copy code
pip install opencv-python tensorflow keras openpyxl pillow ipywidgets numpy
Note
Make sure to set the correct paths for the image directory, Haar Cascade files, and the Excel sheet.
The system requires a pre-trained model for face recognition. You can download the VGGFace model from Kaggle.
The system uses the cosine similarity to match faces. You can adjust the similarity threshold according to your needs.
The system currently supports only .jpg images. If you want to use other formats, you will need to modify the code accordingly.
