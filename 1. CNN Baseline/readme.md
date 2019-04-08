Sequential Model using CNN Layers

<pre>
Instructions to load model in Tensorboard

from keras.callbacks import TensorBoard

Create logs directory on your local drive
Add this code in as well

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

In the “fit” section of the code make sure that TensorBoard is called
 
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=10, batch_size=32,callbacks=[tensorboard])

RUN the code and when the model finishes running go to prompt and put

tensorboard --logdir=path/to/log-directory

Go to the browser and type
Localhost:6006

However I noticed that when you put the path to logs directory the tensorboard will not work. 
The tensor model is actually saved in a higher directory than the logs.

So if your directory structure looks like this:
Root/test1/logs

Your models get saved in test1 and your logs get saved in logs.
So you need to point the TensorBoard to test1

For example:
TensorBoard  —logdir Test1

Then you go to the browser and type localhost:6006
</pre>
