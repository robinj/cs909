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

public class PLSA implements CustomModel {
	
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
	
	private double noOfInstances;
	private int noOfClasses;
		
	public PLSA() {
	
		//LSA specific constructors
		this.lsa = new LatentSemanticAnalysis();
		this.ranker = new Ranker();
		
		//Text specific constructors
		this.swv = new StringToWordVector(); 
		this.rm = new Remove();
		this.fc = new FilteredClassifier();
		this.noOfClasses = 75;
		
	}
	
	public void runFilteredClassifier(Instances data, Instances test, Classifier classifier, String cName) {
		
		//Cross validation fold and random seed
		int folds = 5;
		Random rand = new Random(1);
		
		try {
			
			for (int i = 1; i < this.noOfClasses; i++) {
		
				System.out.println("(RemoveFilter): Selected class attribute " + i);
		
				//**Remove all attribute classes not relevant this pass**
				this.rm.setAttributeIndicesArray(new int[] {0,i});
				this.rm.setInvertSelection(true);
				this.rm.setInputFormat(test);
				Instances removedData = Filter.useFilter(test, this.rm);
				
				//**Apply StringToWordVector filter**
				removedData.setClassIndex(1);
				this.fc.setFilter(this.swv);
				this.fc.setClassifier(classifier);
				System.out.println("(STWFilter): Applied StringToWordVector");
				String[] options = {"-C", "-T", "-L"};
    				StringToWordVector stwv = new StringToWordVector();

    				try {
    					stwv.setOutputWordCounts(true);
    	    				stwv.setOptions(options);
    	    				stwv.setInputFormat(removedData);    
    				}
    				catch(Exception e) {
    	    				System.out.println(e);
    				}
				
				removedData = Filter.useFilter(removedData,stwv);
				removedData = performLSA(removedData);
				
				this.fc.buildClassifier(removedData);
			
				//**Run Classifier**
				System.out.println("(TFIDFModel): Running evaluation of " + cName + " on TFIDF Model");
				Evaluation eval = new Evaluation(removedData);
				eval.evaluateModel(this.fc, removedData);
				//eval.crossValidateModel(this.fc, removedData, folds, rand);
				
				averagedCorrect = averagedCorrect + eval.correct();
				averagedCorrectPct = averagedCorrectPct + eval.pctCorrect();
				averagedIncorrect = averagedIncorrect + eval.incorrect();
				averagedIncorrectPct = averagedIncorrectPct + eval.pctIncorrect();
				averagedRMSE = averagedRMSE + eval.rootMeanSquaredError();
				
				System.out.println("Correctly Classified: " + eval.correct() + " (" + eval.pctCorrect() + "%)");
				System.out.println("Incorrectly Classified: " + eval.incorrect() + " (" + eval.pctIncorrect() + "%)");
				System.out.println("RMSE: " + eval.rootMeanSquaredError());
				
				System.out.println();

				
			}
			
		} catch (Exception err) {
		
			err.printStackTrace(System.out);
		
		}
	}
	
	
	public Instances performLSA(Instances data) {

		//Apply LSA evaluator options
		String[] lsaoptions = {"-N","-A", "-1","-R","10"};
		String[] rankeroptions = {"-T -1.7976931348623157E308","-N 750"};
		
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
	
	//deprecated but interface still contains it so safer to leave for now
	public Instances[] runModel(Instances data, Instances data2) { return null; }
}
