# Deep Learning

Coursework on neural networks, from dense networks and CNNs to RNNs, transformers, GANs, segmentation and model explainability.

All notebooks were run in **Google Colab** and are shown here **with their original outputs**.

## Highlights
- **[Chest X-Ray Pneumonia Classifier](pneumonia_classifier.ipynb)**: binary classification of chest X-rays (normal vs. pneumonia) using ResNet50 transfer learning, with data exploration and evaluation.
- **[Image Segmentation with U-Net](Assignment%209_ImageSegmentation_UNet.ipynb)**: pixel-level segmentation on the Oxford-IIIT Pet dataset.
- **[Grad-CAM Explainability](Introduction_to_GradCam.ipynb)**: visualizes which image regions drive a CNN's predictions.

## Notebooks
| Notebook | Topic |
|---|---|
| [Assignment 1: Iris Flower Classification](Assignment_1_Iris_FLower_Classification.ipynb) | K-Nearest Neighbors with scikit-learn |
| [Assignment 2: MNIST Classification](Assignment_2_MNIST%20Classification.ipynb) | Dense neural network with Keras |
| [Assignment 3: CIFAR-10 with CNNs](Assignment_3_CiFAR_10_Images_Classification_Using_CNNs.ipynb) | Convolutional neural network image classification |
| [Assignment 4: Transfer Learning](Assignment_4_Transfer_Learning_MNIST.ipynb) | Freezing layers and transferring a CNN between digit subsets |
| [Assignment 5: Stock Price Prediction with RNNs](Assignment5_Stock%20Price%20Prediction%20Using%20RNNs.ipynb) | LSTM forecasting of Google stock prices |
| [Assignment 6: Image Compression](Assignment6_ImageCompression.ipynb) | Autoencoder compression of MNIST images |
| [Assignment 8: Sentiment Analysis with BERT](Assignment8_Sentiment_Analysis_with_BERT.ipynb) | Fine-tuning BERT on IMDB reviews |
| [Assignment 9: U-Net Segmentation](Assignment%209_ImageSegmentation_UNet.ipynb) | Semantic segmentation on Oxford-IIIT Pet |
| [Assignment 10: GANs on MNIST](Assignment_10_GANs_MNIST.ipynb) | DCGAN generating handwritten digits |
| [Assignment 11: LIME](Assignment_11_LIME.ipynb) | Explaining text-classifier predictions on 20 Newsgroups |
| [Introduction to Grad-CAM](Introduction_to_GradCam.ipynb) | CNN explainability heatmaps on CIFAR-10 |
| [Pneumonia Classifier](pneumonia_classifier.ipynb) | ResNet50 transfer learning on chest X-rays |

Data files: `Google_Stock_Price_Train.csv`, `Google_Stock_Price_Test.csv`, `trainingandtestdata/testdata.manual.2009.06.14.csv`.

## Datasets not included (too large for GitHub)
- **Chest X-ray images** for the pneumonia classifier: [CoronaHack Chest X-Ray Dataset (Kaggle)](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset)
## Tech stack
Python · TensorFlow/Keras · Hugging Face Transformers · scikit-learn · LIME · pandas · NumPy · Matplotlib · Seaborn

## Notes
- Paths like `/content/drive/...` refer to Google Drive mounted in Colab.
- If a large notebook doesn't preview on GitHub, open it on [nbviewer](https://nbviewer.org/) instead.
