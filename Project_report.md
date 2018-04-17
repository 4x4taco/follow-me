## Udacity Project 3: Perception

### Introduction  
The purpose of the Udacity follow me project was to get the students familiar with buidling and training fully convolutional neural networks for the purpose performing semantic segmenation.  The Semantic segmentation was to locate the hero individual in an environment full of other people.  Once the hero was found the quad copter was supposed to follow the hero target around the environment.  




### Environment Setup
After using git and cloning the RoboND-Segmentation-Lab the contents were explored and the Segementation-Lab.ipynb file was opened in Jupyter Notebook to begin building the FCN.

### AWS Configuration
The most difficult portion of this project was transferring files from my local machine to the AWS instance I had created to speed up trainging of the Neural Network.  I choose the route of setting up a role within AWS.  This role provided full access to my S3 object that contained all of the data for the project.  Once the role was created I had create a new instance that included access to the S3 object.  AWS client tools then had to be installed and the files were copied to the Ubuntu/home directory for easy access.  

```python


```

The next step for exercise 1 was to implement a passthrough filter to eliminate the bottom portion of the table and take as much of the table top away from the point cloud.  This fileter is axis specific with a minimuim and a maximum user specified values which limit the length of the clipping action.  The code to implement this passthrough filter looks like this.

```python

```
I set my values to .6 and 1.1 for the min and max values respectively.  I thought this did the best job of clipping the unwated information from the point cloud.

The next filter applied in exercise 1 one was the RANSAC segmentation.  The purpose of this filter allows the user to identify unique shapes within an environment and then choose to extract the objects identified or the objects that represent the outliers.  The code used to accomplish this task is shown below.

```python

```

Using all three code snippets above the final output is shown below.  These point cloud files were saved locally to the directory I was working in and the image under manipulation was static.  To verify that my code was working correctly I had to call pcl_viewer from the terminal window to see the effects each of the filters had on the original image.

### Ex1 extracted outliers

![](./pics/extracted_outliers.PNG)
