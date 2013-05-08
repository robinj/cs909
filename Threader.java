import java.util.*; //may want to change this later to be more specific
import java.util.List;
import java.util.Random;
import weka.core.Instances;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;

import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.Remove;
import weka.attributeSelection.InfoGainAttributeEval; 
import weka.attributeSelection.Ranker;
import weka.filters.MultiFilter; 

import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;

class Threader {

	Coordinator coord; 
	ArrayList<Thread> models;
	
	
	private double averagedIncorrectPct;
	private double averagedIncorrect;
	private double averagedCorrectPct;
	private double averagedCorrect;
	private double averagedRMSE; 
	
	private Instances data; 
	private Instances test; 
	private Classifier classifier; 
	private DataCollector dc;
	private String cName; 
	
	Threader(Instances data, Instances test, Classifier classifier, String cName) {
		coord = new Coordinator(); 
		models = new ArrayList<Thread>();
		dc = new DataCollector();
		
		this.data = data;
		this.test = test;
		this.classifier = classifier; 
		this.cName = cName; 
	}
	
	public synchronized void execute() {
		
		
		//iterate over each label attribute
		for(int i = 1; i <76; i++) {
			//wait for a free worker
			while(coord.isBusy()) {
				try{
					System.out.println("waiting");
					synchronized(dc) {
						dc.wait();
					}
				} catch (Exception e) {
					e.printStackTrace(); 
				}
			}
			
			//create a new model (for the specific attribute class)
			PLSA model = new PLSA(i);
			//set some parameters
			model.runFilteredClassifier(data,  test,  classifier,  cName);
			//set the monitors (needed for threading)
			model.setMonitors(coord, dc);
	
			//dispatch the thread and stick it in the "thread pool"
			Thread myThread = new Thread(model);
			myThread.start();
			models.add(myThread);
			
			//tell coordinator that this thread is now dispatched 
			synchronized(coord) {
				coord.dispatchWorker();
			}			
			System.out.println("Despatched new worker thread(" + i + ")");			
			
		}
		
		//wait for all results to come in before displaying them
		while(coord.isWorking()) {
			try{
				System.out.println("waiting");
				synchronized(dc) {
					dc.wait();
				}
			} catch (Exception e) {
			}
		}
		
		System.out.println("Average correct:   " + dc.getCorrect()/75);
		System.out.println("Average incorrect: " + dc.getIncorrect()/75);
		
		//calculate percentages etc here 
	}
	
	
	
	


}