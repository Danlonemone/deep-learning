# Deep Learning

Coursework on neural networks, from dense networks and CNNs to RNNs, transformers, GANs, segmentation and model explainability.

All notebooks were run in **Google Colab** and are shown here **with their original outputs**.

## Highlights
- **[Chest X-Ray Pneumonia Classifier](pneumonia_classifier.ipynb)**: binary classification of chest X-rays (normal vs. pneumonia) using two-phase ResNet50 transfer learning with class weighting. Scored 94% on validation but 77% on the held-out test set; [results and analysis below](#pneumonia-classifier-results).
- **[Image Segmentation with U-Net](Assignment%209_ImageSegmentation_UNet.ipynb)**: pixel-level segmentation on the Oxford-IIIT Pet dataset.
- **[Grad-CAM Explainability](Introduction_to_GradCam.ipynb)**: visualizes which image regions drive a CNN's predictions.

## Pneumonia classifier results

ResNet50 pretrained on ImageNet, trained in two phases: a new classification head on a frozen base, then fine-tuning of the last residual block (`conv5`). Class weights offset the roughly 3:1 pneumonia-to-normal imbalance. The 5,286 training images were split 85/15 into train and validation; the dataset's separate 624-image test set was used only for the final evaluation.

| | Accuracy | ROC-AUC |
|---|---|---|
| Validation (793 images, split from the training pool) | 94.3% | 0.977 |
| **Test (624 images, separate set)** | **77.4%** | **0.883** |

Per class on the test set (threshold 0.5):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.897 | **0.449** | 0.598 |
| Pneumonia | 0.746 | 0.969 | 0.843 |

**What went wrong.** The model catches 97% of pneumonia cases but labels more than half of the normal X-rays as pneumonia (it predicted pneumonia for 507 of 624 test images; 390 actually were). Moving the decision threshold to 0.3 or 0.7 barely changes the result, so this isn't a calibration problem: the model hasn't learned what a normal X-ray looks like in the test set. The large gap between validation and test points to a distribution shift. Validation images were drawn from the same pool as training, while the test set is a separate collection, so validation accuracy overstated how well the model generalizes.

**What I'd do next:** build a validation set that resembles the test set instead of a random slice of the training pool, add augmentation (contrast, cropping, rotation) so the model can't rely on scanner- or source-specific cues, and use Grad-CAM to check whether the model is looking at the lungs or at artifacts such as labels and borders.

## Notebooks
| Notebook | Topic |
|---|---|
| [Assignment 1: Iris Flower Classification](Assignment_1_Iris_FLower_Classification.ipynb) | K-Nearest Neighbors with scikit-learn |
| [Assignment 2: MNIST Classification](Assignment_2_MNIST%20Classification.ipynb) | Dense neural network with Keras |
| [Assignment 3: CIFAR-10 with CNNs](Assignment_3_CiFAR_10_Images_Classification_Using_CNNs.ipynb) | Convolutional neural network image classification |
| [Assignment 4: Transfer Learning](Assignment_4_Transfer_Learning_MNIST.ipynb) | Freezing layers and transferring a CNN between digit subsets |
| [Assignment 5: Stock Price Prediction with RNNs](Assignment5_Stock%20Price%20Prediction%20Using%20RNNs.ipynb) | LSTM forecasting of Google stock prices |
| [Assignment 6: Image Compression](Assignment6_ImageCompression.ipynb) | Autoencoder compression of MNIST images |
| [Assignment 9: U-Net Segmentation](Assignment%209_ImageSegmentation_UNet.ipynb) | Semantic segmentation on Oxford-IIIT Pet |
| [Assignment 10: GANs on MNIST](Assignment_10_GANs_MNIST.ipynb) | DCGAN generating handwritten digits |
| [Assignment 11: LIME](Assignment_11_LIME.ipynb) | Explaining text-classifier predictions on 20 Newsgroups |
| [Introduction to Grad-CAM](Introduction_to_GradCam.ipynb) | CNN explainability heatmaps on CIFAR-10 |
| [Pneumonia Classifier](pneumonia_classifier.ipynb) | ResNet50 transfer learning on chest X-rays |

Data files: `Google_Stock_Price_Train.csv`, `Google_Stock_Price_Test.csv`.

## Datasets not included (too large for GitHub)
- **Chest X-ray images** for the pneumonia classifier: [CoronaHack Chest X-Ray Dataset (Kaggle)](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset)

## Tech stack
Python · TensorFlow/Keras · scikit-learn · LIME · pandas · NumPy · Matplotlib · Seaborn

## Notes
- Paths like `/content/drive/...` refer to Google Drive mounted in Colab.