class DataCollector {
	
	private double correct;
	private double incorrect; 
	
	DataCollector() {
		this.correct = 0;
		this.incorrect = 0; 
	}
	
	public synchronized void addCorrect(double c) {this.correct += c;}
	public synchronized void addIncorrect(double i) {this.incorrect += i;}
	
	public synchronized double getCorrect() {return this.correct;}
	public synchronized double getIncorrect() {return this.incorrect;}

	


}