## Udacity Project 4: Follow me

![](./pics/intro_img.PNG)
### Introduction  
The purpose of the Udacity follow me project was to get the students familiar with buidling and training a fully convolutional neural network for the purpose of performing semantic segmenation.  The semantic segmentation objective was to locate the hero individual in an environment full of other people.  Once the hero was found the quad copter was supposed to follow the hero target around the environment.  


### Environment Setup
After using git and cloning the RoboND-Segmentation-Lab the contents were explored and the Segementation-Lab.ipynb file was opened in Jupyter Notebook to begin building the FCN.  After the code had been tested the functions used for the segmentaion labe were copied to the deep learning lab.  The simulation was run several more times with various hyper parameters to obtain the highest accuracy.

### AWS Configuration
The most difficult portion of this project was transferring files from my local machine to the AWS instance, I had created to speed up training of the Neural Network.  I choose the route of setting up a role within AWS.  This role provided full access to my S3 object that contained all of the data for the project.  Once the role was created I had to create a new instance that included access to the S3 object.  AWS client tools then had to be installed and the files were copied to the Ubuntu/home directory for easy access.  

### Encoder
The encoder is essentially a fully convolutional network that takes an input image and applies a series of convolutions to reduce the resolution to a lower and more compact scale.  When the resolution and scale are at the desired values a set of 1x1 convolutions can be applied to do pixel by pixel evaluations looking for any object that a user may be trying to recognize.  The object of interest will have a unique value when processed by the 1x1 convolutions 

### Decoder
The Decoder is essentially the opposite of the Encoder.  The Decoder takes the output from the 1x1 convolutions and upscales them to their original size.  This is a strength of FCN's, the fact that the input image size does not matter for the training of the network.  Billinear upsampling was used for this project.  This means that for the 2 x 2 filter the upsampling to a 4x4 would be linearly interpolated between the values.  layer concatenation was implemented in the decoder block to improve accuracy.  The layer concatenation works by taking an unsampled layer with a larger amount of spatial information and adding to it pixel by pixel filtered information. 
### Bilinear upsampling
![](./pics/lin_int.PNG)

### FCN layer construction
3 functions were provided by Udacity to ease the burden of creating all of the layers for the FCN.  The the seperable_conv2d_batchnorm function was used in both the Encoder and Decoder blocks.  The strides were were set to two with same padding.  The functions both used relu activations which are easy to implement and allow a controllable non-linearity to be applied to the system, this allows the model to approximate more complex problems. A seperable 2d convolution with batch normalization was used via the kera module.  Batch normalization is the process of normalizing the data at each input step to the layers.  This helps to keep the numbers in a range that plays nice for numberical computing.  Seperable convolutions differ from normal convolutions in the way that pixels are mapped.  Seperable convolutions perofrm both a depthwise convolution and point by point convolution.  The settings for these convolutions are shown below in the code section with the inputs to the functions being (inputs/input_layer, depth of filter, strides).  The decoder layers have the same input.  

```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    layer1 = encoder_block(inputs, 64, 2)
    layer2 = encoder_block(layer1, 128, 2)    
        
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    layer3 = conv2d_batchnorm(layer2, 256, kernel_size=1, strides=1)
   
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    layer4 = decoder_block(layer3, layer1, 64)
    layer5 = decoder_block(layer4, inputs, 128)
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(layer5)
```

```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```
### FCN diagram
![](./pics/fcn_diag1.PNG)
The heirarch of the FCN constructed started with a a 256x256x3 rgb image.  This image was passed through a convolutional filter of 3x3 with a stride of two that reduced the height and width to: layer1 = 128x128, layer2=64x64 the coresponding decoder layers are the same height and width.  The 1x1 convolutions had a depth of 256.  It was thought that having a higher number for the 1x1 convolutions would result in the model being able to differentiate more precisely.  The 1x1 conolutions in the center section of this model play an important role of making this FCN a better classifier by increasing the depth without changine the structure too much.  Th 1x1 conolutions also consume a lower amout of computing resources.  1x1 conolutions map each pixel to a filter and essentially transform the area of an image to a filter that is deep while reducing dimensionality.  1x1 convolutions were choosen over a fully connected layer to reduce computational burden but increase accuracy.  A fully connected layer means that each neuron is connected to every other neuron, this would increase computational effort.  Increasing the amout of computation would be due to the extra weights and the back propagation error that would need to be computed at each connection.

### FCN Model and Parameters
An Encoder and Decoder function was created to aid in the setup of the individaul layers.  Using these functions individual layers were created.  The order of the layers starts with the first layer after the input image and extends to layer 5.  Layer 5 is positioned just before the ouput of the FCN.    
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
first run on AWS segmentaion lab
learning_rate = .001
batch_size = 50
num_epochs = 50
steps_per_epoch = 200
validation_steps = 50
workers = 20
approximate time = 227s * 50epochs= 189 min
```

```python
second runon on AWS segmentation lab
learning_rate = .001
batch_size = 50
num_epochs = 50
steps_per_epoch = 50
validation_steps = 50
workers = 20
approximate time = 69s * 50 epochs = 57 min 
```

```python
final run on AWS Deep learning lab
learning_rate = .0007
batch_size = 50
num_epochs = 40
steps_per_epoch = 100
validation_steps = 50
workers = 20
approximate time = 187s * 100 epochs = 311 min 
```
The configuration of hyper parameters between the two runs of the segmentaiton lab had a large difference in the time required to compute.  The accuracy of the second run for the segmentaion lab was higher even for a reduced training time.  I had to decrease the learning rate to .0007 and increase the steps per epoch for the final simulation to reach an accuracy of 44%.  

Listed above are the Hyper parameters used to train the FCN for Semantic Segmentaion.  The first two represent the inputs for the segmentation lab and the last one is the final inputs for the Deep learning lab.  The learning rate is a multiplier that is used to increase or decrease the values of the weights used for learning.  This learning rate is multiplied by the backpropagation error formed from the chain rule.  I choose a low value to increase my overal accuracy.  I could have choosen a larger value but desire accuracy over speed.  If a larger value was choosen the model would reach steady state accuracy faster but then over and under shoot the target after reaching its critcal point.

The batch size corresponds the the number of images or samples that are input into the network per pass.  I Choose a value of 50 to ensure the batch size was small compared to the overall data set. this reduces the amount of memory occupied during the computations.  When testing I receieved resources exhausted errors when my batch size was set to a value of over 80. 

Num epochs was set to 50 for the segmentation lab.  This is the number of loops that network will go though.  the number or steps per epoch must be completed before the next epoch can start.  This means 50 pictures per step were input for 200 steps and 50 epochs.  50*200*50 = 500,000 samples.  This value was changed to 100 on the final model to ensure an accuracy of > 40% was acheived.

Steps per epoch represent the number of times the batch size is input into the newtork per epoch.  Changing this value has a signifigant increase on time required to train but only a margnial increase on the accuracy.  For the final simulation the steps per epoch was changed to 100. I was trying to increase the number of pictures fed through the newtork without increasing the number of epochs.

Workers represent the number of individual processes the platform can operate simultaneously.  This is analgous to thinking about men performing some time of construction/renovation.  The more workers a project has the faster it can be accomplished but the tougher it can be to organize.  If there is slight deviation to the orginaization the step can be performed out of order and cause signifigant set backs to the project.  Since the cpu is responsible for the coordination it can be taxing on the system if too many workers are called.  I choose a number of 20 which represents approximately 5 workers per cpu.  Given the high performance of the AWS machine I didnt think it was too high.

### Model training and accuracy
### Following Target
![](./pics/following-target.PNG)

### Patrolling without Target
![](./pics/patrolling_without_target.PNG)

### Patrolling with Target
![](./pics/patrolling_with_target.PNG)

### 1st run loss vs epochs segmentation lab
![](./pics/1st%20run.PNG)

### Final run loss vs epochs segmentation lab
![](./pics/training_curves.PNG)


## Conclusion
Utilizing the AWS system I was able to train my model on the images provided in the repository.  I was able to get my accuracy up to 44%  by tweeking the parameters for the Deep Learning lab.  I tried several times to implement my trained model in the quad copter environment, to test the follower function.  I tried implementig on two different machines with the RoboND environemnt installed, the error below was received on both machines.  I ensured that the directions were correctly followed and that the correct dependencies had been installed to support the simulator.  I used AWS s3 buckets to transfer files to an from my AWS instance and local directories.  I tested this feature on a known file type to ensure that there was'nt some type of conversion error and it worked perfectly.  I did not see any common problems for the error I received and don't know what could be causing it. The trained model did an excellent job identifying the target at close range this should be expected given the way a FCN operates.  When the hero is close the reduced resolution still does a good job of retaining the pixel information, but at large distance the reduced resolution indicates many false negatives, only one false positive and 152 true positives instead of 539 true positives while following closely. 

Pictures of people were used to train this FCN to look for a specific person, but images of almost any object could be detected given the right training set.  These images could be cars, animals, types of plants, or even medical imagery looking for anomalies.

### Future enhancements
Moving forwared I would like to get the simulator to work correctly and capture more data with the hero in the environent at a distance, to implement a higher accruracy for my model.  Some items to improve the accuracy would be additional photos of the hero in the environment.  This data could be used to further train the model.  Lowering the learning rate would also improve the accuracy.  Looking at the Final loss vs epoch the train loss and val loss begin to seperate with more epochs.  This is refered to as overfitting.  Keeping the epochs at 75 and lowering the learning rate would produce a more accurate model.
### Final accuracy
![](./pics/error_final.PNG)

### Final loss vs epoch
![](./pics/training_curve_final.PNG)

### Sources
1)  https://www.quora.com/What-is-the-difference-between-an-usual-convolution-layer-and-a-separable-convolution-layer-used-in-depthwise-separable-DL-models

2)  https://en.wikipedia.org/wiki/Convolutional_neural_network

3)  various Udacity notes and code
