Project Introduction:

This repository contains a comprehensive suite of computer vision tools integrated into a user-friendly PyQt5 interface. It includes various modules such as ResNet50 for binary classification, VGG19 for numeric classification trained on the MNIST dataset, PCA for image reconstruction, optical flow analysis, and background subtraction for video processing. These tools are designed to demonstrate advanced computer vision techniques and facilitate their understanding through practical application.

Getting Started:

To run the project, execute the main.py file. This will launch the PyQt5 interface, as shown in the provided screenshots. Through this interface, you can interact with the various functionalities of the project.

Testing the Functionalities:

1.Background Subtraction and Optical Flow: Load the videos from their respective folders within the Datasets directory to test these features.

2.PCA Dimension Reduction: Load images for PCA analysis from the corresponding folders in the Datasets directory, or use any other images you wish to test.

3.Binary Classification with ResNet50 (Dog and Cat): Similar to PCA, load the necessary images from the Datasets folder designed for this task.

4.Numeric Classification with VGG19: Enter a digit between 0 and 9 in the black screen on the interface and press the 'Predict' button. To test another number, click the 'Reset' button.

Additional Features:

1.Model Structures and Training Graphs: Use the provided buttons in the interface to view the structures of the VGG19 and ResNet50 models and their training graphs.

Note: The model should be trained by the user.
 
### Required Software and Versions
- **Python**: `3.11.5`
- ** PyTorch **: `2.1.0+cu118`
- **CUDA**: `11.7`

Conclusion
This suite is intended as a practical exploration tool for students and professionals interested in deepening their knowledge of computer vision applications. It provides hands-on experience with real-world data and visualization of the model's internal workings.