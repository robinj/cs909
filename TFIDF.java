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

/*TFIDF Model with Multiple Classifier Implementations
  - Uses FilteredClassifier
*/

public class TFIDF extends Thread implements CustomModel {
	
	private StringToWordVector swv; 
	private Remove rm;
	private FilteredClassifier fc;
	private double averagedIncorrectPct;
	private double averagedIncorrect;
	private double averagedCorrectPct;
	private double averagedCorrect;
	private double averagedRMSE;
	
	private MultiFilter mf;
	private AttributeSelection as; 
	private Reorder ro; 
	private Reorder rro;
	private InfoGainAttributeEval igae;
	private Ranker rkr; 
	
	private double noOfInstances;
	private int noOfClasses;
	
	private boolean complete; 
	private int classID; 
	
	private Instances data; 
	private Instances test; 
	private Classifier classifier; 
	private String cName; 
	
	private Coordinator coord; 
	private DataCollector datacollect; 
	
	public TFIDF (int classID){
		this.complete = false; 
		
		this.swv = new StringToWordVector(); 
		this.rm = new Remove();
		this.fc = new FilteredClassifier();
		
		this.as = new AttributeSelection(); 
		this.ro = new Reorder(); 
		this.rro = new Reorder(); 
		this.igae = new InfoGainAttributeEval();
		this.rkr = new Ranker(); 
		this.mf = new MultiFilter();

		this.classID = classID; 		
		this.noOfClasses = 75;
		
	}
	
	public void runFilteredClassifier(Instances data, Instances test, Classifier classifier, String cName) {
		this.data = data;
		this.test = test;
		this.classifier = classifier; 
		this.cName = cName; 
	}
	
	public void setMonitors(Coordinator co, DataCollector dc) {
		this.datacollect = dc; 
		this.coord = co; 
	}
	
	public synchronized void run(){
		//Apply StringToWordVector TFIDF Option
		String swvoptions[] = {"-W 2000", "-I", "-L", "-M 1"};
		
		//Apply the AtributeSelection options
		String asOptions[] = {
		
		};
		
		//Apply the Reorder options
		String roOptions[] = {
			"-R last-first"
		};
		
		//Apply the ReReorder options
		String rroOptions[] = {
			"-R last-first"
		};
		
		String agaeOptions[] = {
		
		};
		
		//Apply Ranker options
		String rkrOptions[] = {
			"-T -1.7976931348623157E308",
			"-N 750"
		};
		
		//Cross validation fold and random seed
		int folds = 5;
		Random rand = new Random(1);
		
		try {	
			//Set input format and options for sw and filteredclassifier
			this.swv.setInputFormat(data);
			this.swv.setOptions(swvoptions);
			this.fc.setClassifier(classifier);
			
			//set input format for Reorder 
			this.ro.setInputFormat(data);
			this.ro.setOptions(roOptions);
			
			//set input format for Reorder
			this.rro.setInputFormat(data);
			this.rro.setOptions(rroOptions);
			
			//seet input format for Ranker
			this.rkr.setOptions(rkrOptions);
			
			//set input format for AttributeSelection
			//this.as.setInputFormat(data);
			//this.as.setOptions(asOptions);
			this.as.setEvaluator(igae);
			this.as.setSearch(rkr); 
			
			//Remove all attribute classes
			this.rm.setAttributeIndicesArray(new int[] {0,classID});
			this.rm.setInvertSelection(true);
			this.rm.setInputFormat(data);
			Instances removedData = Filter.useFilter(data, this.rm);
			
			//set all of the options for the MultiFilter
			Filter[] filters = {this.swv, this.ro, this.as, this.rro};
			this.mf.setFilters(filters);
			
			//Apply StringToWordVector filter
			removedData.setClassIndex(1);
			this.fc.setFilter(this.mf);
			System.out.println("(STWFilter): Appled StringToWordVector");
			
			//Build classifier on filtered data
			this.fc.buildClassifier(removedData);
		
			//Present results and store averages
			System.out.println("(TFIDFModel): Running evaluation of " + cName + " on TFIDF Model");
			Evaluation eval = new Evaluation(removedData);
			//eval.evaluateModel(this.fc, removedData);
			eval.crossValidateModel(this.fc, removedData, folds, rand);
			
			//give data to the datacollector 
			datacollect.addCorrect(eval.correct());
			datacollect.addIncorrect(eval.incorrect());
			
		} catch (Exception err) {
		
			err.printStackTrace(System.out);
		
		}
		
		//tell the coordinator that this thread's work is done 
		coord.endWorker();
		
		synchronized(datacollect){
			datacollect.notifyAll();
		}
	}
	
	
	public boolean isComplete() {
		return complete; 
	}
	
	
	/*
	*	Potentially deprecated method
	*	Do not change at present.
	*/
	
	public Instances[] runModel(Instances is, Instances is2) {
		
		//wordvector instances
		Instances[] wordVectors = new Instances[2];
		
		//Set filter options
		String swvOptions[] = {"-W 100","-I","-L", "-M 1"};
		
		System.out.println("(TFIDF Model): Applying...");
		
		try {
		
			System.out.println("Instances before processing: " + is.firstInstance());
			
			is.setClassIndex(1);
			this.swv.setInputFormat(is);
					
			//Set attribute indices
			this.swv.setAttributeIndices("0");
			
			//Do not operate on per class basis
			this.swv.setWordsToKeep(100);
			
			this.swv.setOptions(swvOptions);
		
			wordVectors[0] = this.swv.useFilter(is, this.swv);
			wordVectors[1] = this.swv.useFilter(is2, this.swv);
			
			System.out.println("(TFIDF Model): Text Model Created");
			
			System.out.println("Instances after processing: " + wordVectors[0].firstInstance());

		
		} catch (Exception err) {
		
			System.out.println("(TFIDF Model)(Error): " + err.toString());
		
		}
		
		return wordVectors;
		
	}
	
	

}