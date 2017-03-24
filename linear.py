import tensorflow as tf
import numpy as np

# create session
sess = tf.Session()

# 저수준 API 사용하기
x = tf.placeholder(tf.float32)
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(W))
print(sess.run(b))

y = tf.placeholder(tf.float32)

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)

for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))

