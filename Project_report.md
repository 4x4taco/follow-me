## Udacity Project 3: Perception

### Introduction  
The purpose of the Udacity follow me project was to get the students familiar with buidling and training fully convolutional neural networks for the purpose performing semantic segmenation.  The Semantic segmentation was to locate the hero individual in an environment full of other people.  Once the hero was found the quad copter was supposed to follow the hero target around the environment.  




### Environment Setup
After using git and cloning the RoboND-Segmentation-Lab the contents were explored and the Segementation-Lab.ipynb file was opened in Jupyter Notebook to begin building the FCN.

### AWS Configuration
The most difficult portion of this project was transferring files from my local machine to the AWS instance I had created to speed up trainging of the Neural Network.  I choose the route of setting up a role within AWS.  This role provided full access to my S3 object that contained all of the data for the project.  Once the role was created I had create a new instance that included access to the S3 object.  AWS client tools then had to be installed and the files were copied to the Ubuntu/home directory for easy access.  

### Encoder
The encode is essentially a fully convolutional network that takes an input image and applies a series of convolutions to reduce the resolution to a lower and more compact scale.  When the resolution and scale are at the desired values a set of 1x1 convolutions can be applied to do pixel by pixel evaluations looking for any object that a user may be trying to recognize.  The object of interest will have a unique value when processed by the 1x1 convolutions 

### Decoder
The Decoder is essentially the opposite of the Encoder.  The Decoder takes the output from the 1x1 convolutions and upscales them to their original size.  This is a strength of FCN's, the fact that the input image size does not matter for the trainign of the network.  Billinear upsampling was used for this project.  This means that for the 2 x 2 filter the upsampling to a 4x4 would be linearly interpolated between the values.  layer concatenation was implemented in the decoder block to improve accuracy.  The layer concatenation works by taking an unsampled layer with a larger amount of spatial information and adding to it pixel by pixel the filtered information. 

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

The heirarch of the FCN constructed started with a a 256x256x3 rgb image.  This image was passed through a convolutional filter of 3x3 with a stride of two that reduced the height and width to ????? and the depth to 64.  The next convolutional filter reduced the height and width to ???? and the depth was increased to 128.  The 1x1 convolutions had a depth of 256.  It was thought that having a higher number for the 1x1 convolutions that the accruacy would be increase becuase the depth could recognize 256 unique values.  The decoder side of the FCN is th exact opposite of the Encoder side.

### FCN Model and Parameters
An Encoder and Decoder function were created to aid in the setup of the individaul layers.  Using these functions individual layers were created.  The order of the layers starts with the first layer after the input image and extends to layer 5.  Layer 5 is positioned just before the ouput of the FCN.    
```python
def encoder_block(input_layer, filters, strides):
    
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides=2)
    
    return output_layer
    
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    up_sample = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([up_sample, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    
    t_layer = separable_conv2d_batchnorm(concat_layer, filters, strides=1)
    output_layer = separable_conv2d_batchnorm(t_layer, filters, strides=1)
    return output_layer
```

### Hyper parameters and the Meaning
```python
learning_rate = .001
batch_size = 50
num_epochs = 50
steps_per_epoch = 200
validation_steps = 50
workers = 20
```

Listed above are the Hyper parameters used for to train the FCN for Semantic Segmentaion.  The learning rate is a multiplier that is used to increase or decrease the values of the weights used for learning.  This learning rate is multiplied by the backpropagation error formed from the chain rule.  I choose a low value to increase my overal accuracy.  I could have choosen a larger value but desire accuracy over speed.  If a larger value was chooosen the model would reach steady state accuracy faster but then over and under shoot the target after reaching its critcal point.

The batch size corresponds the the number of images or samples that are input into the network per pass.  I Choose a value of 50 to ensure the batch size was small compared to the overall data set.  50*12*10^3 bytes = 

Num epochs was set to 50.  This is the number of loops that network will go though.  the number or steps per epoch must be completed before the next epoch can start.  This means 50 pictures per step were input for 200 steps and 50 epochs.  50*200*50 = 500,000 samples.

Steps per epoch represen the number of times the batch size is input into the newtork per epoch.  Changing this value has a signifigant increase on time required to train but only a margnial increase on the accuracy.

Workers represent the number of individual processes the platform can operate simultaneously.  This is analgous to thinking about men performing some time of construction/renovation.  The more workers a project has the faster it can be accomplished but the tougher it can be to organize.  If there is slight deviation to the orginaization the step can be performed out of order and cause signifigant set backs to the project.  Since the cpu is responsible for the coordination it can be taxing on the system if too many workers are called.  I choose a number of 20 which represents approximately 5 workers per cpu.  Given the high performance of the AWS machine I didnt think it was too high.

### Model training and accuracy

### Ex1 extracted outliers

![](./pics/extracted_outliers.PNG)
