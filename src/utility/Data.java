package utility;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;

public class Data {
	public int[][][] U;
	public Token[] tokens;
	public Random rand;

	public Data(int I, int J, int R, Random rand) {
		this.rand = rand;
		U = new int[I][J][R];
		for(int i=0; i<I; i++){
			for(int j=0; j<J; j++){
				for(int r=0; r<R; r++){
					U[i][j][r] = -1;
				}
			}
		}
		tokens = new ArrayList<Token>().toArray(new Token[0]);
	}
	
	public void readBowData(String dataFile) throws IOException{
		ArrayList<Token> tlist = new ArrayList<Token>();
		BufferedReader br = MyUtil.Reader(dataFile);
		while(true){
			String str = br.readLine();
			if(str == null) break;
			StringTokenizer st = new StringTokenizer(str, ",");
			int j = Integer.valueOf(st.nextToken());
			int i = Integer.valueOf(st.nextToken());
			int w = Integer.valueOf(st.nextToken());
			int freq = Integer.valueOf(st.nextToken());
			for(int f=0;f<freq;f++){
				tlist.add(new Token(j, i, w));
			}
		}
		br.close();
		tokens = tlist.toArray(new Token[0]);
	}

	public void readRatingData(String dataFile) throws IOException{
		BufferedReader br = MyUtil.Reader(dataFile);
		while(true){
			String str = br.readLine();
			if(str == null) break;
			StringTokenizer st = new StringTokenizer(str, ",");
			int j = Integer.valueOf(st.nextToken());
			int i = Integer.valueOf(st.nextToken());
			int r = Integer.valueOf(st.nextToken());
			U[i][j][r] = Integer.valueOf(st.nextToken());
		}
		br.close();
	}
}
