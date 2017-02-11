

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
    public Signal(int tab[]){
        int N = tab.length;
        data = new double[N];
        for(int i=0;i<N;i++){
            data[i] = tab[i];
        }
        Fe = 44100;
        samples = N;
        ajust_2();
    }
    public Signal(int tab[], int F_ech){
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
    public double get(int i){
        return data[i];
    }
    public void set(int i, int s){
        data[i] = s;
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
}
