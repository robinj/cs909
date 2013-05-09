import java.util.List;

import weka.attributeSelection.LatentSemanticAnalysis;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Classifier;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.Evaluation;
import java.util.Random;

public class PLSA extends Thread implements CustomModel {
	
	private LatentSemanticAnalysis lsa;
	private Ranker ranker;
	private StringToWordVector swv;
	private Remove rm;
	private FilteredClassifier fc;
	
	private double averagedIncorrectPct;
	private double averagedIncorrect;
	private double averagedCorrectPct;
	private double averagedCorrect;
	private double averagedRMSE; 
	
	private int classID; 
	
	private double noOfInstances;
	private int noOfClasses;
	
	private Instances data; 
	private Instances test; 
	private Classifier classifier; 
	private String cName;

	private Coordinator coord; 
	private DataCollector datacollect; 		
		
	public PLSA(int classID) {
	
		//LSA specific constructors
		this.lsa = new LatentSemanticAnalysis();
		this.ranker = new Ranker();
		
		//Text specific constructors
		this.swv = new StringToWordVector(); 
		this.rm = new Remove();
		this.fc = new FilteredClassifier();
		this.noOfClasses = 75;
		this.classID = classID; 
		
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
	
	public void run (){
		
		//Cross validation fold and random seed
		int folds = 5;
		Random rand = new Random(1);
		
		try {
	
		/* FILTER TRAINING DATA AND BUILD CLASSIFIER */

			//System.out.println("(RemoveFilter): Selected class attribute " + i);
	
			//**Remove all attribute classes not relevant this pass**
			
			this.rm.setAttributeIndicesArray(new int[] {0,classID});
			this.rm.setInvertSelection(true);
			this.rm.setInputFormat(data);
			Instances removedData = Filter.useFilter(data, this.rm);
			
			//**Apply StringToWordVector filter**
			
			removedData.setClassIndex(1);
			this.fc.setFilter(this.swv);
			this.fc.setClassifier(classifier);
			System.out.println("(STWFilter): Applied StringToWordVector");
			String[] options = {"-C", "-T", "-L"};
				StringToWordVector stwv = new StringToWordVector();

			//**Set string to word vector options**

				try {
					stwv.setOutputWordCounts(true);
						stwv.setOptions(options);
						stwv.setInputFormat(removedData);    
				}
				catch(Exception e) {
						System.out.println(e);
				}
			
			//**Perform LSA on removed training data**

			removedData = Filter.useFilter(removedData,stwv);
			removedData = performLSA(removedData);
			
			this.fc.buildClassifier(removedData);

			/* FILTER TEST DATA AND RUN CLASSIFIER ON TEST SET */
			
			//**Apply StringToWordVector filter**
			//**Remove all attribute classes not relevant this pass**

							this.rm.setAttributeIndicesArray(new int[] {0,classID});
							this.rm.setInvertSelection(true);
							this.rm.setInputFormat(test);
							Instances removedTestData = Filter.useFilter(test, this.rm);

							removedData.setClassIndex(1);
							this.fc.setFilter(this.swv);
			this.fc.setClassifier(classifier);
			System.out.println("(STWFilter): Applied StringToWordVector");
							String[] optionsTest = {"-C", "-T", "-L"};
							StringToWordVector stwvTest = new StringToWordVector();

							//**Set	string to word vector options**

							try {
									stwv.setOutputWordCounts(true);
									stwv.setOptions(optionsTest);
									stwv.setInputFormat(removedTestData);
							}
							catch(Exception e) {
									System.out.println(e);
							}

							//**Perform LSA	on removed training data**

							removedTestData = Filter.useFilter(removedTestData,stwv);
							removedTestData = performLSA(removedTestData);


			//**Run Classifier**
			System.out.println("(PLSAModel): Running evaluation of " + cName + " on PLSA Model");
			Evaluation eval = new Evaluation(removedTestData);
			eval.evaluateModel(this.fc, removedTestData);
			//eval.crossValidateModel(this.fc, removedTestData, folds, rand);
			
			averagedCorrect = averagedCorrect + eval.correct();
			averagedCorrectPct = averagedCorrectPct + eval.pctCorrect();
			averagedIncorrect = averagedIncorrect + eval.incorrect();
			averagedIncorrectPct = averagedIncorrectPct + eval.pctIncorrect();
			averagedRMSE = averagedRMSE + eval.rootMeanSquaredError();
			
			System.out.println("Correctly Classified: " + eval.correct() + " (" + eval.pctCorrect() + "%)");
			System.out.println("Incorrectly Classified: " + eval.incorrect() + " (" + eval.pctIncorrect() + "%)");
			System.out.println("RMSE: " + eval.rootMeanSquaredError());
			
			System.out.println();
			
			//update the data collector
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
	
	
	public Instances performLSA(Instances data) {

		//Apply LSA evaluator options
		String[] lsaoptions = {"-N","-A", "-1","-R","750"};
		String[] rankeroptions = {"-T -1.7976931348623157E308","-N 250"};
		
		try 
        	{   
            		lsa.setOptions(lsaoptions);
            		System.out.println("Options set");
            		data.setClassIndex(0);
            		System.out.println("Class index set");
            		lsa.buildEvaluator(data);
            		System.out.println("Evaluator built");
            		ranker.setOptions(rankeroptions);
            		ranker.search(lsa, lsa.transformedData(data));
            		System.out.println("ranker search done");
            		return lsa.transformedData(data);
        
		} catch(Exception e) {
            		System.out.println(e);
       		}
    
    	return null;
    
	}
	
	public boolean isComplete() {
		return false; 
	}
	
	//deprecated but interface still contains it so safer to leave for now
	public Instances[] runModel(Instances data, Instances data2) { return null; }
	
	}

