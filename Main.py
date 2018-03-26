from NNBrain import Brain
import numpy

SamplesDir = 'Samples.txt'
TestResultDir = 'TestResult.txt'
X_Dimenson = 4
Y_Dimenson = 1
Train_Steps = 10000

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

	samples = max_min_normalize(samples,0)
	samples = z_score_normalize(samples,1)
	samples = z_score_normalize(samples,2)
	samples = z_score_normalize(samples,3)


	brain = Brain(x_dimenson = X_Dimenson,y_dimenson = Y_Dimenson,sample_size = len(samplesLines))
	brain.store_samples(samples)



	for i in range(0,Train_Steps,1):
		brain.learn()
		print('已完成步数：'+str(i))

	error_sum = 0

	brain.plot_loss()

	test_count = 100

	result_file = open(TestResultDir,mode = 'w+')

	for i in range(1,test_count,1):
		testInput = numpy.array([samples[i,:X_Dimenson]])
		result = brain.test_result(testInput)
		error_sum +=numpy.abs(result-samples[i,X_Dimenson])
		result_file.writelines(str(float(result[0]))+'\n')
		print('第'+str(i)+'个测试结果为：' + str(result))

	error_aver = error_sum/test_count

	print('平均误差：'+str(error_aver))

	result_file.close()




def max_min_normalize(samples,column_index):
	tmpMax = numpy.max(samples[:,column_index])
	tmpMin = numpy.min(samples[:,column_index])
	for i in range(0,len(samples),1):
		samples[i,column_index] = (samples[i,column_index]-tmpMin)/(tmpMax-tmpMin)

	return samples

def z_score_normalize(samples,column_index):
	tmpAver = numpy.average(samples[:,column_index])
	tmpStd = numpy.std(samples[:,column_index])

	for i in range(0,len(samples),1):
		samples[i,column_index]=(samples[i,column_index]-tmpAver)/tmpStd

	return samples

if __name__ == "__main__":
	run_mainLoop()
