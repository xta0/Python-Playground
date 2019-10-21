import tensorflow as tf

hello = tf.constant("Hellow Tensorflow")
sess = tf.compat.v1.Session()
print(sess.run(hello))
