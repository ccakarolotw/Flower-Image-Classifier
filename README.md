# Flower-Image-Classifier
An image classifier application based on Deep Convolutional Neural Network

##Task
Build an AI application that can recognize different species of flowers. 

##Summary
The application is built with Pytorch. It includeds functions to load and preprocess the image files. The classifier is built on top of pretrained neural networks resnet18. The test accuracy is around 90%.  

##Run Locally
<ul><li>Download train.py, predict.py, load_data.py, predict_utility.py, handle_command_line.py, cat_to_name.json  </li>
  <li>Train the model by running train.py <code>python train.py data_directory -epoch=20</code></li>
  <li>Use the application to predict flower species based on image by running predict.py <code>python predict.py /path/to/image checkpoint</code></li>
</ul>
