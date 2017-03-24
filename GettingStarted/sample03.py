import tensorflow as tf

a = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = a * x + b

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
# (h(x) - y) ^ 2
squared_deltas = tf.square(linear_model - y)
# 비용함수 만듦
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([a, b]))