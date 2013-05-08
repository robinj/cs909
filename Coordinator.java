class Coordinator {

	private int cores;
	private int workers; 
	
	private int notifications; 
	
	Coordinator() {
		this.workers = 0;
		this.notifications = 0; 
	
		this.cores = Runtime.getRuntime().availableProcessors();
	}
	
	public synchronized boolean isWorking() {
		return this.workers!=0; 
	}
	
	public int getCores() {
		return this.cores; 
	}
	
	public synchronized void dispatchWorker() {
		this.workers++;
	}
	
	public synchronized void endWorker() {
		this.workers--;
	}
	
	public synchronized boolean isBusy() {
		return (this.cores == workers);
	}
	
	public synchronized boolean isNotifying() {
		return !(this.notifications == 0);
	}
	
	public synchronized void sendNotification() {
		this.notifications++; 
		notifyAll(); 
	}
	
	public synchronized void clearNotification() {
		this.notifications--; 
	}
}