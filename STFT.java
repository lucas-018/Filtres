
public class STFT {
	int T;
	int frame;
	int hop;
	Complex data[];
	int init_samples;
	public STFT(double tab[], int in_frame, int in_hop) {
		frame = 1;
		while(frame<in_frame) {
			frame *= 2;
		}
		hop = in_hop;
		int N = tab.length;
		init_samples = N;
		T = 0;
		int av = 0;
		while(av<N-frame) {
			av += hop;
			T += 1;
		}
		data = new Complex[T*frame];
		for(int t=0;t<T;t++) {
			Complex x[] = new Complex[frame];
			for(int i=0;i<frame;i++) {
				x[i] = new Complex(tab[i+t*hop], 0);
				x[i] = x[i].scale(0.5-0.5*Math.cos(2*Math.PI*i/frame));
			}
			Complex X[] = FFT.fft(x);
			for(int i=0;i<frame;i++) {
				set(t, i, X[i]);
			}
		}
	}
	public int frame() {
		return frame;
	}
	public int num() {
		return T;
	}
	public int hop() {
		return hop;
	}
	public int initSamples() {
		return init_samples;
	}
	public Complex get(int n_time, int n_freq) {
		return data[n_time*frame+n_freq];
	}
	public void set(int n_time, int n_freq, Complex z) {
		data[n_time*frame+n_freq] = new Complex(z.re, z.im);
	}
	public double[] istft() {
		double tab[] = new double[init_samples];
		for(int i=0;i<init_samples;i++) {
			tab[i] = 0;
		}
		for(int i=0;i<T;i++) {
			Complex X[] = new Complex[frame];
			for(int j=0;j<frame;j++) {
				X[j] = new Complex(get(i, j).re, get(i, j).im);
			}
			Complex x[] = FFT.ifft(X);
			for(int j=0;j<frame;j++) {
				tab[i*hop+j] += x[j].re;
			}
		}
		return tab;
	}
}
