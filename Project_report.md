## Udacity Project 3: Perception

### Introduction  
The purpose of the Udacity follow me project was to get the students familiar with buidling and training fully convolutional neural networks for the purpose performing semantic segmenation.  The Semantic segmentation was to locate the hero individual in an environment full of other people.  Once the hero was found the quad copter was supposed to follow the hero target around the environment.  




### Environment Setup
After using git and cloning the RoboND-Segmentation-Lab the contents were explored and the Segementation-Lab.ipynb file was opened in Jupyter Notebook to begin building the FCN.

### AWS Configuration
The most difficult portion of this project was transferring files from my local machine to the AWS instance I had created to speed up trainging of the Neural Network.  I choose the route of setting up a role within AWS.  This role provided full access to my S3 object that contained all of the data for the project.  Once the role was created I had create a new instance that included access to the S3 object.  AWS client tools then had to be installed and the files were copied to the Ubuntu/home directory for easy access.  

### Encoder
The encode is essentially a fully convolutional network that takes an input image and applies a series of convolutions to reduce the resolution to a lower and more compact scale.  When the resolution and scale are at the desired values a set of 1x1 convolutions can be applied to do pixel by pixel evaluations looking for any object that a user may be trying to recognize.  

### Decoder
The Decoder is essentially the opposite of the Encoder.  The Decoder takes the output from the 1x1 convolutions and upscales them to their original size.  This is a strength of FCN's, the fact that the input image size does not matter for the trainign of the network.  Billinear upsampling was used for this project.  This means that for the 2 x 2 filter the upsampling to a 4x4 would be linearly interpolated between the values.

### Upsampling Interpolation

![](./pics/lin_int.PNG)

### FCN layer construction
3 functions were provided by Udacity to ease the burden of creating all of the layers for the FCN.  The the seperable_conv2d_batchnorm function was used in both th Encoder and Decoder blocks.  The strides were were set to two with same padding which means that the image after applying the kernel and strides that the output will remain the same size after the convolutions are applied.  The functions both used relu activations which are easy to implement and allow a controllable non-linearity to be applied to the system allowing the model to computer more complex problems.  The bilinear_upsample function was used in the Decoder block.  
```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

```

```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```


```python

```

Using all three code snippets above the final output is shown below.  These point cloud files were saved locally to the directory I was working in and the image under manipulation was static.  To verify that my code was working correctly I had to call pcl_viewer from the terminal window to see the effects each of the filters had on the original image.

### Ex1 extracted outliers

![](./pics/extracted_outliers.PNG)
