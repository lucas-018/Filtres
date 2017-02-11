import java.io.File;
import java.io.IOException;
import java.io.DataInputStream;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.SourceDataLine;

public class MainClass {
	public static byte[] getSamples(String filePath) throws IOException {
		try {
			File in_file = new File(filePath);
			AudioInputStream stream = AudioSystem.getAudioInputStream(in_file);
			if(stream == null) {
				throw new IllegalArgumentException("There is no AudioStream to process");
			}
			int length = (int)(stream.getFrameLength()*stream.getFormat().getFrameSize());
			byte samples[] = new byte[length];
			DataInputStream inputSamples = new DataInputStream(stream);
			inputSamples.readFully(samples);
			return samples;
		}catch(Exception Ex) {
			byte res[] = new byte[1];
			return res;
		}
	}
	public static void main(String[] args){
        /*double tab[] = new double[100000];
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
        System.out.println(x.length);*/
		/*int total_frames = 0;
		File in_file = new File("C:/Users/Lucas/Enr_3.wav");
		try {
			AudioInputStream audioIn = AudioSystem.getAudioInputStream(in_file);
			int bytes_per_frame = audioIn.getFormat().getFrameSize();
			if(bytes_per_frame == AudioSystem.NOT_SPECIFIED) {
				bytes_per_frame = 1;
			}
			System.out.println(bytes_per_frame);
			int numBytes = 1024*bytes_per_frame;
			byte audioBytes[] = new byte[numBytes];
			try {
				int numBytesRead = 0;
				int numFramesRead = 0;
				while((numBytesRead = audioIn.read(audioBytes)) != -1) {
					numFramesRead = numBytesRead/bytes_per_frame;
					total_frames += numFramesRead;
				}
			}catch(Exception ex2){
				//Exeption_2
			}
		}catch(Exception e1) {
			//Exception_1
		}*/
		byte samples[] = new byte[1];
		try {
			samples = getSamples("C:/Users/Lucas/Enr_3.wav");
		}catch(Exception e) {
			//
		}
		for(int i=0;i<samples.length;i++) {
			System.out.println(samples[i]);
		}
    }
}
