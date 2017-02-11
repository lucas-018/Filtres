

public class Signal {
    double data[];
    int samples;
    int Fe;
    int size_2;
    private void ajust_2(){
        size_2=1;
        while(size_2<samples) {
            size_2 *= 2;
        }
    }
    public Signal(){
        data = new double[1];
        data[0] = 0;
        samples = 1;
        Fe = 44100;
        size_2 = 2;
    }
    public Signal(int in_samples) {
    	data = new double[in_samples];
    	samples = in_samples;
    	Fe = 44100;
    	ajust_2();
    }
    public Signal(double tab[]){
        int N = tab.length;
        data = new double[N];
        for(int i=0;i<N;i++){
            data[i] = tab[i];
        }
        Fe = 44100;
        samples = N;
        ajust_2();
    }
    public Signal(double tab[], int F_ech){
        int N = tab.length;
        data = new double[N];
        for(int i=0;i<N;i++){
            data[i] = tab[i];
        }
        Fe = F_ech;
        samples = N;
        ajust_2();
    }
    public int size(){
        return size_2;
    }
    public int samples(){
        return samples;
    }
    public void setSamples(int in_samples) {
    	samples = in_samples;
    }
    public void setFe(int F_ech) {
    	Fe = F_ech;
    }
    public double get(int i){
        return data[i];
    }
    public void set(int i, int s){
        data[i] = s;
    }
    public Signal copy() {
    	Signal S = new Signal(data, Fe);
    	return S;
    }
    public double moyenne() {
    	double moy = 0;
    	for(int i=0;i<samples;i++) {
    		moy += data[i];
    	}
    	moy = moy/samples;
    	return moy;
    }
    public double variance() {
    	double var = 0;
    	for(int i=0;i<samples;i++) {
    		var += data[i]*data[i];
    	}
    	var = var/samples;
    	double moy = moyenne();
    	var -= moy*moy;
    	return var;
    }
    public Complex[] fft(){
        Complex x[] = new Complex[size_2];
        Complex X[] = new Complex[size_2];
        for(int i=0;i<size_2;i++){
        	
            if(i<samples){
            	x[i] = new Complex(data[i], 0);
            }else {
                x[i] = new Complex(0,0);
            }
        }
        X = FFT.fft(x);
        return X;
    }
    public void passeHaut(double f_c, double alpha) {
    	Complex X[] = fft();
    	int N = X.length;
    	int C = (int)(N*f_c/Fe);
    	for(int i=1;i<C+1;i++) {
    		X[i] = X[i].scale(Math.pow((i/C), alpha));
    		X[N-i] = new Complex(X[i].re, X[i].im);
    	}
    	Complex y[] = FFT.ifft(X);
    	for(int i=0;i<samples;i++) {
    		data[i] = y[i].re;
    	}
    }
    public void passeBas(double f_c, double alpha) {
    	Complex X[] = fft();
    	int N = X.length;
    	int C = (int)(N*f_c/Fe);
    	for(int i=C;i<(int)(N/2)+1;i++) {
    		X[i] = X[i].scale(Math.pow((((int)(N/2)-i)/((int)(N/2)-C)), alpha));
    		X[N-i] = new Complex(X[i].re, X[i].im);
    	}
    	Complex y[] = FFT.ifft(X);
    	for(int i=0;i<samples;i++) {
    		data[i] = y[i].re;
    	}
    }
    public void smooth(int power) {
    	double x[] = new double[samples];
    	double val = 0;
    	for(int i=0;i<samples;i++) {
    		x[i] = data[i];
    	}
    	for(int j=0;j<2*power+1;j++) {
    		val += x[j];
    	}
    	for(int i=power;i<samples-power;i++) {
    		data[i] = val/(2*power+1);
    		if(i<samples-power-1) {
    			val -= x[i-power];
    			val += x[i+power+1];
    		}
    	}
    }
    public void cutEcart(double alpha) {
    	double sigma = Math.sqrt(variance());
    	STFT stft = new STFT(data, 1024, 512);
    	for(int i=0;i<stft.num();i++) {
    		for(int j=0;j<stft.frame();j++) {
    			if(stft.get(i,  j).abs()>alpha*sigma) {
    				stft.set(i,  j,  0);
    			}
    		}
    	}
    	data = stft.istft();
    }
}
