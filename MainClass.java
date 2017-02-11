

public class MainClass {
	public static void main(String[] args){
        double tab[] = new double[100000];
        for(int i=0;i<100000;i++){
            tab[i] = (5*Math.sin((float)i/5.0));
        }
        //Signal S = new Signal(tab);
        //Complex X[] = S.fft();
        STFT stft = new STFT(tab, 1024, 512);
        double x[] = stft.istft();
        for(int i=0;i<10000;i++){
            //System.out.println(X[i].re + ", " + X[i].im);
        	//System.out.println(tab[i]);
        	System.out.println(x[i]-tab[i]);
        }
    }
}
