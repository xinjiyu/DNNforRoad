# NNBrain

'''
// ┏┛ ┻━━━━━┛ ┻┓
// ┃　　　　　　 ┃
// ┃　　　━　　　┃
// ┃　┳┛　  ┗┳　┃
// ┃　　　　　　 ┃
// ┃　　　┻　　　┃
// ┃　　　　　　 ┃
// ┗━┓　　　┏━━━┛  Code is far away from bug with the animal protecting
//   ┃　　　┃   神兽保佑
//   ┃　　　┃   代码无BUG！
//   ┃　　　┗━━━━━━━━━┓
//   ┃　　　　　　　    ┣┓
//   ┃　　　　         ┏┛
//   ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
//     ┃ ┫ ┫   ┃ ┫ ┫
//     ┗━┻━┛   ┗━┻━┛
'''

import tensorflow as tf
import numpy
import matplotlib.pyplot as plot

Batch_Size = 10
InputLayerUnits = 10
HiddenLayerUnits = 5
OutputLayerUnits = 1
LearningRate = 0.01


class Brain(object):

	def __init__(self, x_dimenson, y_dimenson,sample_size):
		self.x_dimenson = x_dimenson
		self.y_dimenson = y_dimenson
		self.sample_size = sample_size
		self.samples = numpy.zeros((self.sample_size, self.x_dimenson + self.y_dimenson))
		self.learnSteps = 0
		self.loss_list = []

		self.activation = tf.nn.relu
		self.sess = tf.Session()

		self.input_x = tf.placeholder(tf.float32, [None, self.x_dimenson], 'Input_X')
		self.real_y = tf.placeholder(tf.float32, [None, self.y_dimenson], 'Real_Y')

		self.calc_y = self.build_net()
		self.build_trainOp()

		self.sess.run(tf.global_variables_initializer())

	def build_net(self):
		w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
		with tf.variable_scope('Network'):
			inputLayer = tf.layers.dense(inputs = self.input_x, units = InputLayerUnits, activation = self.activation,
			                             kernel_initializer = w_initializer, bias_initializer = b_initializer,
			                             name = 'InputLayer')

			hiddenLayer = tf.layers.dense(inputs = inputLayer, units = HiddenLayerUnits, activation = self.activation,
			                              kernel_initializer = w_initializer, bias_initializer = b_initializer,
			                              name = 'HiddenLayer')

			outputLayer = tf.layers.dense(inputs = hiddenLayer, units = OutputLayerUnits, activation = self.activation,
			                              kernel_initializer = w_initializer, bias_initializer = b_initializer,
			                              name = 'OutputLayer')

			return outputLayer

	def build_trainOp(self):
		with tf.variable_scope('Loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.real_y,self.calc_y))

		with tf.variable_scope('TrainOp'):
			self.trainOp = tf.train.AdamOptimizer(LearningRate).minimize(self.loss)

	def store_samples(self, samples):
		self.samples = samples

	def learn(self):
		sample_index = numpy.random.choice(len(self.samples), size = Batch_Size)
		batch_samples = self.samples[sample_index, :]
		tmp, loss = self.sess.run([self.trainOp, self.loss],
		                          feed_dict = {self.input_x: batch_samples[:, :self.x_dimenson],
		                                       self.real_y: batch_samples[:, self.x_dimenson: ]})
		self.learnSteps +=1
		self.loss_list.append(loss)

	def test_result(self,input):
		result = self.sess.run(self.calc_y,feed_dict = {self.input_x:input[:]})
		return result


	def plot_loss(self):
		plot.plot(numpy.arange(len(self.loss_list)),self.loss_list)
		plot.ylabel('Loss')
		plot.xlabel('Train Steps')
		plot.show()

