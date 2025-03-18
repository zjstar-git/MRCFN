## A Study of Microseismic Timing Prediction via a Multimodal Response Complementary Fusion

This code implements a multimodal response complementary fusion (MRCFN) microseismic indicator prediction network for coal mine underground. Based on the microseismic physical index characteristics and microseismic spatial distribution information of the working face, the multimodal information is enhanced by using the multi-attention module for the preliminary fusion of feature information, and the fusion of multimodal feature information is achieved by fusing the output feature maps through the vector dot product fusion to achieve the prediction and analysis of microseismic indexes. The performance of MRCFN is analysed by comparison with Informer, CNN and LSTM.

## Environment
This code was developed and tested in the following environment:

Python: 3.9
Keras: 2.2(2.0 or later versions are recommended)
Tensorflow: 2.6.0
CUDA: 11.2

## Required Dependencies
Install the required dependencies for the model via requirements.txt:
pip install -r requirements.txt

## Data preparation
Microseismic physical metrics: index_data.py
Output: create_data.py
Model main program: main.py


## License
This project is licensed under the MIT License. See the LICENSE file for more details.