

public class MainClass {
	public static void main(String[] args){
        int tab[] = new int[1000];
        for(int i=0;i<1000;i++){
            tab[i] = (int)(5*Math.sin((float)i/5.0));
        }
        Signal S = new Signal(tab);
        Complex X[] = S.fft();
        for(int i=0;i<X.length;i++){
            System.out.println(X[i].re + ", " + X[i].im);
        	//System.out.println(tab[i]);
        }
        System.out.println(S.size());
    }
}
