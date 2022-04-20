# dogs_classification

# Dog breed detection from images with convolutional neural networks and transfert learning

This is a simple flask/html website who take a picture from dogs and predict the breeds

Project with flask , using keras and Xception to predict dog breeds

- Pre-process the images with specific techniques (e.g. Whitening, equalization, possibly modification of the size of the images).
- Perform data augmentation (mirroring, cropping ...).
- Use the transfer learning and thus use an already trained network.
- Fine-tuning of the pre-trained model


Data
The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. It was originally collected for fine-grain image categorization, a challenging problem as certain dog breeds have near identical features or differ in colour and age.

Number of categories: 120
Number of images: 20,580
Annotations: Class labels, Bounding boxes
Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011. [pdf] [poster] [BibTex]

Secondary: J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009. [pdf] [BibTex]


![image](https://user-images.githubusercontent.com/74118071/163820751-23155bb8-6b28-4204-ad26-27eef4895f8d.png)
