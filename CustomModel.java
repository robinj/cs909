import weka.core.Instances;
import weka.classifiers.Classifier;

interface CustomModel{
	
	Instances[] runModel(Instances data, Instances data2); 
	void runFilteredClassifier(Instances trainingData, Instances testData, Classifier classifier, String cName);
	boolean isComplete(); 
	void run();

}