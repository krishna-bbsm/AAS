# Deep Learning-Based Attendance Tracking System
# Description
The Deep Learning-Based Attendance Tracking System leverages the power of Deep learning, computer vision and one-shot learning method for recognizing individuals from a single sample image and automates the process of attendance marking.The system uses transfer learning by building the model upon the pre-trained VGGFace model, extending its capabilities to suit the specific needs of an attendance system, offering the capability to detect and recognize multiple faces in an image. The recognized faces are then cross-referenced with an existing database, and the attendance is marked with a timestamp in an Excel sheet.

# Dependencies
- OpenCV
- TensorFlow
- Keras
- openpyxl
- PIL
- ipywidgets
- numpy

# Important Notes
- Image Collection: The system will automatically mount your Google Drive and set up the working directory. It will then collect images from the specified directory and extract faces from them.
- Face Processing: The extracted faces are processed and their representations are computed using the VGGFace model.
- Image Capture: The system will capture an image using the webcam. The captured image is processed, and faces in the image are detected.
- Face Recognition: The detected faces are recognized by comparing their representations with the ones computed earlier.
- Attendance Marking: If a match is found, the system will mark the attendance of the recognized person in an Excel sheet with the current timestamp.


## Appendix

This section provides additional information and resources that can help you understand and use the Automated Attendance System more effectively.

- Face Recognition:
    Face recognition is a method of identifying or verifying the identity of an individual using their face. It captures, analyzes, and compares patterns based on the person's facial details. The face detection process is an essential and crucial step in face recognition, and it's used to locate human faces in images. This process involves identifying the features (eyes, nose, mouth, etc.) and the geometry of the face.

-  VGGFace Model:
    VGGFace is a pre-trained deep learning model that is specifically designed for face recognition tasks. It's based on the VGG16 architecture and trained on a large dataset of faces. The model can be used to generate a vector representation (embedding) of a face, which can then be compared to the embeddings of other faces to determine their identity.

- Code:
    The code for this project is written in Python and uses the TensorFlow and Keras libraries to implement the deep learning model. The OpenCV library is used for image processing tasks, and the openpyxl library is used to write the attendance data to an Excel file.

- Troubleshooting:

    If you encounter any issues while running the project, here are a few things you can try:
    - Make sure all the dependencies are installed correctly.
    - If running on Google Colab, make sure the runtime is connected. please note that Google Colab does not support access to the webcam for capturing images. As a result, you will need to provide the input images manually.
    - If you're running the project locally and the webcam capture isn't working, make sure your Python environment has access to your webcam.
## Features

- Face Recognition
- Multi-Face Handling
- Versatile Recognition (Can recognize both frontal and profile faces)
- Automated Attendance Marking (Marks attendance with a timestamp in an Excel sheet automatically.)

## Output

![Detected faces](https://github.com/krishna-bbsm/Deep-Learning-Based-Attendance-Tracking-System/blob/3ba1426a26e5f232e3907f44f85286ec9266d78a/output_image.jpg)


![Attendace Sheet](Attendance_Sheet.xlsx)

## Built

[![Code: Python](https://img.shields.io/badge/Made%20with-Python-3776AB.svg)](https://shields.io/)

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Development Environment

This project is developed and run in Google Colab, a cloud-based Jupyter notebook environment that provides access to computing resources, but  also it can be modified to run on a local environment.
    
## Documentation

[Documentation](https://github.com/krishna-bbsm/Automated_Attendance_System/blob/main/README.md)


## Authors

- [@KrishnaBandaru](https://www.github.com/octokatherine)





# Hi, I'm Krishna Bandaru! 👋



If you have any feedback, 
please reach out to me at krishna.bbsm@gmail.com

