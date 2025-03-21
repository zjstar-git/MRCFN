## A Study of Microseismic Timing Prediction via a Multimodal Response Complementary Fusion

This code implements a multimodal response complementary fusion (MRCFN) microseismic indicator prediction network for coal mine underground. Based on the microseismic physical index characteristics and microseismic spatial distribution information of the working face, the multimodal information is enhanced by using the multi-attention module for the preliminary fusion of feature information, and the fusion of multimodal feature information is achieved by fusing the output feature maps through the vector dot product fusion to achieve the prediction and analysis of microseismic indexes. The performance of MRCFN is analysed by comparison with Informer, CNN and LSTM.

## Environment
This code was developed and tested in the following environment:

Python: 3.9<br>
Keras: 2.6.0<br>
Tensorflow: 2.6.0<br>
CUDA: 11.2<br>

## Required Dependencies
Install the required dependencies for the model via requirements.txt:
```python
pip install -r requirements.txt
```

## Data preparation
Microseismic physical metrics: index_data.py <br>
Output: create_data.py<br>
Model main program: main.py<br>


## License
This project is licensed under the MIT License. See the LICENSE file for more details.