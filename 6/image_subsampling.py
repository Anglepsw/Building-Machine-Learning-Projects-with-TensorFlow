
# coding: utf-8

# In[35]:


import tensorflow as tf
import numpy as np

#Generate the filename queue, and read the gif files contents
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("data/test.gif"))
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
image=tf.image.decode_gif(value)

#Define the  coordinator
coord = tf.train.Coordinator()

def normalize_and_encode (img_tensor):
    image_dimensions = tf.shape(img_tensor.eval()[0]).eval()
    return tf.image.encode_jpeg(tf.reshape(tf.cast(img_tensor, tf.uint8), image_dimensions))

with tf.Session() as sess:
    maxfile=open ("maxpool.jpeg", "wb+")
    avgfile=open ("avgpool.jpeg", "wb+")
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(coord=coord)
    
    image_tensor = tf.image.rgb_to_grayscale(sess.run([image])[0])
    image_gray=np.array(image_tensor.eval()[0][:][:])
    maxed_tensor=tf.nn.avg_pool(tf.cast(image_tensor, tf.float32),[1,2,2,1],[1,2,2,1],"SAME")
    a=maxed_tensor.eval()
    averaged_tensor=tf.nn.avg_pool(tf.cast(image_tensor, tf.float32),[1,2,2,1],[1,2,2,1],"SAME")
    b=averaged_tensor.eval()

    maxfile.write(normalize_and_encode(maxed_tensor).eval())
    avgfile.write(normalize_and_encode(averaged_tensor).eval())
    coord.request_stop()
    maxfile.close()
    avgfile.close()
coord.join(threads)


