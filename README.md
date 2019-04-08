How to Use:

Download Dataset by Sully Chen: [https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view] Size: 25 minutes = 25{min} x 60{1 min = 60 sec} x 30{fps} = 45,000 images ~ 2.3 GB

Note: You can run without training using the pretrained model if short of compute resources

<b>Use python3 Predict_model_v2.py to run the model on the dataset</b>


<pre>
To visualize training using Tensorboard use tensorboard 

from keras.callbacks import TensorBoard
Create logs directory on your local drive
Add this code in as well

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

In the “fit” section of the code make sure that TensorBoard is called
  
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=10, batch_size=32,callbacks=[tensorboard])

RUN the code and when the model finishes running go to prompt and put

tensorboard --logdir=path/to/log-directory

when you put the path to logs directory the tensorboard will not work. The tensor model is actually saved 
in a higher directory than the logs.

So if your directory structure looks like this:

Root/test1/logs

Your models get saved in test1 and your logs get saved in logs.

So you need to point the TensorBoard to test1

For example:

TensorBoard  —logdir Test1

Then you go to the browser and type localhost:6006
</pre>

<b>
Other Larger Datasets you can train on
</b>

(1) Udacity: https://medium.com/udacity/open-sourcing-223gb-of-mountain-view-driving-data-f6b5593fbfa5
70 minutes of data ~ 223GB
Format: Image, latitude, longitude, gear, brake, throttle, steering angles and speed

(2) Udacity Dataset: https://github.com/udacity/self-driving-car/tree/master/datasets [Datsets ranging from 40 to 183 GB in different conditions]

(3) Comma.ai Dataset [80 GB Uncompressed] https://github.com/commaai/research

(4) Apollo Dataset with different environment data of road: http://data.apollo.auto/?locale=en-us&lang=en

<b>
Some other State of the Art Implementations
</b>

Implementations: https://github.com/udacity/self-driving-car

Blog: https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c

<b>
Credits & Inspired By
</b>

(1) Research paper: End to End Learning for Self-Driving Cars by Nvidia. [https://arxiv.org/pdf/1604.07316.pdf]

(2) Research paper: Self Driving Car Steering Angle Prediction based on Image Recognition by Stanford. [http://cs231n.stanford.edu/reports/2017/pdfs/626.pdf]

(3) Data Source : https://github.com/SullyChen/Autopilot-TensorFlow

(4) https://github.com/cyanamous/Self-Driving-Car-

(5) https://github.com/akshaybahadur21/Autopilot

(6) Nvidia blog: https://devblogs.nvidia.com/deep-learning-self-driving-cars/ 

(7) https://devblogs.nvidia.com/explaining-deep-learning-self-driving-car/
