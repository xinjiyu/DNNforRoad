from NNBrain import Brain
import numpy

SamplesDir = 'Samples.txt'
X_Dimenson = 4
Y_Dimenson = 1
Train_Steps = 500

def run_mainLoop():
	sampleFile = open(SamplesDir)
	samplesLines = sampleFile.readlines()
	sampleFile.close()

	samples = numpy.zeros((len(samplesLines),X_Dimenson+Y_Dimenson))

	for i in range(0,len(samplesLines),1):
		tmpLine = samplesLines[i]
		tmpLine = tmpLine.replace('\n', '')
		tmpCells = tmpLine.split('\t')
		for j in range(0,X_Dimenson+Y_Dimenson,1):
			samples[i,j] = float(tmpCells[j])

	brain = Brain(x_dimenson = X_Dimenson,y_dimenson = Y_Dimenson,sample_size = len(samplesLines))
	brain.store_samples(samples)

	for i in range(0,Train_Steps,1):
		brain.learn()
		print('已完成步数：'+str(i))

	brain.plot_loss()

	testInput = numpy.array([[4.945205479,6661.191781,1608.675342,7.4]])

	result = brain.test_result(testInput)

	print('测试结果为：'+str(result))



if __name__ == "__main__":
	run_mainLoop()
