import java.util.List;

import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.bayes.NaiveBayesMultinomial;

public class TFIDF implements CustomModel {
	
	private StringToWordVector swv; 
	
	public TFIDF (){
	
		this.swv = new StringToWordVector();  
	}
	
	public void runModel(Instances is) {
		
		//Set filter options
		String swvOptions[] = {"-W 1000","-I","-L"};
		
		System.out.println("(TFIDF Model): Applying...");
		
		try {
		
			this.swv.setInputFormat(is);
			this.swv.setOptions(swvOptions);
		
			Instances wordVectors = this.swv.useFilter(is, this.swv);
			
			System.out.println("\n\n(TFIDF Model): Word Vectors -\n\n " + wordVectors);

		
		} catch (Exception err) {
		
			System.out.println("(TFIDF Model)(Error): " + err.toString());
		
		}
		
	}	

}