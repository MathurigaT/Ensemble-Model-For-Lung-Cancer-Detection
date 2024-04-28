# A Deep Learning Approach for Lung Cancer Detection using CT Scan Images

### Abstract

The objective of this project to improve the accuracy and reliability of lung cancer diagnosis by using an ensemble deep learning model based on CT scan pictures. Lung cancer is a significant threat to health around the world, and early detection is critical for improving patient outcomes. While deep learning has showed promise in medical image analysis, ensemble techniques which combine numerous models have the potential to increase performance even more. This paper provides a complete ensemble approach that combines various deep learning architectures with fine tuning and transfer learning and data augmentation strategies to improve lung cancer classification from CT scan images.

### DataSet

The used CT scan images are sourced from a benchmark dataset which is publicly accessible on "The Cancer Imaging Archive" (TCIA) platform.
These CT images along with annotations provided by four experienced radiologists. Personal patient information, including headers, has been removed from DICOM format and converted to JPG format for the purpose of model training, validation, and testing.This dataset has two labels: Malignant and Benign. Entire dataset has beem devided into Test, Val and Train which contains JPEG images with Malignant and Benign labels.

DataSet Link: https://link.springer.com/article/10.1007/s00521-023-09130-7

### Data Pre-Processing

Following combinations were used to augment the images inorder to reduce the class imbalance of training data.
1. Horizontal flip, Contrast adjustment and fill missing pixels with the nearest value.
2. Vertical flip, Contrast adjustment and fill missing pixels with the nearest value.
3. Rotation by 90-degree, Contrast adjustment and fill missing pixels with the nearest value.

![](/Users/mathurigathavarajah/Downloads/DataPreProcessing-DataPreProcessing.jpg)

### Methodology

1. Base model selection

Following pre-trained models were trained with pre-trained wieghts of IMAGENET dataset by modifyng classification layer. Further, fine-tuning technique was applied with different level
of layers.

ResNet Variants - [ResNet50](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/ResNet50_fine_tune.ipynb), [ResNet101](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/ResNet101_fine_tune.ipynb)
DenseNet Variants - [DenseNet121](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/Densenet121_fine_tune.ipynb), [DenseNet169](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/Densenet169_fine_tune.ipynb), [DenseNet201](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/Densenet201_fine_tune.ipynb)
AlexNet- [Modified version of AlexNet](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/Alexnet_fine_tune.ipynb)
VGG Variants - [VGG16](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/VGG16_fine_tune.ipynb)
[InceptionV3](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/Inceptionv3_fine_tune.ipynb)
[EfficientNetB0](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/EfficientNetB0_fine_tune.ipynb)
[MobileNet](https://github.com/MathurigaT/Ensemble-Model-For-Lung-Cancer-Detection/blob/main/MobileNet_fine_tune.ipynb)

From above, the best three base models were selected based on high accuracy and less possibility of over-fitting. As per the results,
ResNet101, DenseNet169 and VGG16 were selected as base model to build an ensemble.

2. Techniques for ensemble model

Following ensemble techniques were used for ensemble model evaluation

Averaging
Soft voting based on combined rankings of accuracy and over-fitting
Hard Voting
Stacking (Random Forest Classifier)
Support Vector Machine (SVM) classifier

### Evaluation

As per the experimented result, Soft voting based on combined rankings of accuracy (94%) and over-fitting shows the most suitable 
ensemble model. Following illustrates the high level architecture of the model.

![](/Users/mathurigathavarajah/Downloads/DataPreProcessing-HighLevelArchitecture.drawio (2).png)

### How to Run the IPython Notebook File

#### Prerequisites

Ensure Python is installed on your system.
Install Jupyter Notebook using `pip install jupyterlab`.

#### Running the Notebook

Clone or download this repository.
Open a terminal and navigate to the directory containing the IPython Notebook file.
Run jupyter notebook to start the server.
Access the notebook at http://localhost:8888/tree.
Open the notebook file and run cells using Shift + Enter.

Refer to [Jupyter documentation](https://docs.jupyter.org/en/latest/) for more details.

Acknowledgement

Department of Computer Science at Cardiff School of Technologies, Cardiff Metropolitan University, for providing me with the opportunity to undertake this research endeavour.
A special acknowledgment goes to [Dr. Sandeep Singh Sengar](https://github.com/sandeepsinghsengar) who is a Senior Lecturer in Computer Science at Cardiff Metropolitan University, for his consistent support, guidance, and constructive feedback throughout this research.
