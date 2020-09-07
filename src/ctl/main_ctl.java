package ctl;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;

import model.Estimation;
import model.Model;
import utility.MTRandom;
import utility.MyUtil;

public class main_ctl {
	final int I = 4, J = 34, R = 10, V = 664, K = 5;
	final String Dir = "data/";
	final String OutDir = "output/";
	final int digit = 10;

	private int T;
	private Random rand;
	private HashMap<String, Integer> mcmcSettings;

	main_ctl(int T) throws IOException {
		this.T = T; 
		rand = new MTRandom();
		mcmcSettings = new HashMap<String, Integer>();
		mcmcSettings.put("MaxMCMC", 40000);
		mcmcSettings.put("burnIn",  20000);
		mcmcSettings.put("interbal", 100);
		mcmcSettings.put("collapsedGibbsInterbal", 10);
	}
	
	public void run() throws IOException {
		Model model = new Model(I, J, R, T, V, K, rand);
		// set data
		model.data.readRatingData(Dir + "rating.csv");
		model.data.readBowData(Dir + "token.csv");

		// run Estimation
		new Estimation(model, rand, mcmcSettings, false, Dir);

		// print parameters
		PrintWriter pw = MyUtil.Writer(OutDir + "T" + T + "_parameters.csv");
		pw.println("alphaI" + "," + MyUtil.doublesTostring(model.alpha_i, digit));
		pw.println("betaI" + "," + MyUtil.doublesTostring(model.beta_i, digit));
		pw.println("alphaR" + "," + MyUtil.doublesTostring(model.alpha_r, digit));
		pw.println("betaR" + "," + MyUtil.doublesTostring(model.beta_r, digit));
		for(int k=0;k<K;k++){
			String line = "dRK" + k + ",";
			for(int r=0;r<R;r++){
				line += MyUtil.form(model.rho_rk[r][k], digit);
				if(r != R-1) line += ",";
			}
			pw.println(line);
		}
		pw.println("theta" + "," + MyUtil.doublesTostring(model.theta, digit));		
		for (int t = 0; t < T; t++) {
			pw.println("wordDist" + t + "," + MyUtil.doublesTostring(model.TopicWordDist[t], digit));
		}
		pw.println("weight_mu" + "," + MyUtil.doublesTostring(model.weight_t, digit));
		pw.println("weight_sigma" + "," + MyUtil.doublesTostring(model.weight_s, digit));
		for (int j = 0; j < J; j++) {
			for (int i = 0; i < I; i++) {
				pw.println("topicDist_j" + j + "_i" + i + "," + MyUtil.doublesTostring(model.DocTopicDist[i][j], digit));
			}
		}
		pw.close();		
	}
	
	private static class ICallable implements Callable<Integer> {
		private main_ctl ct;
		public ICallable(main_ctl ct) {
			this.ct = ct;
		}
		@Override
		public Integer call() throws Exception {
			ct.run();
			return 0;
		}
	}
	
	public static void main(String[] args) throws IOException {
		List<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>();
		for(int T=1; T <= 5; T++) {
			tasks.add(new ICallable(new main_ctl(T)));			
		}			
		MyUtil.startThread(tasks, 5);
	}
}
