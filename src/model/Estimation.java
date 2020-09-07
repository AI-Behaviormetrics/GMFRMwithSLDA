package model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import org.uncommons.maths.random.GaussianGenerator;

import utility.MyUtil;
import utility.Token;

public class Estimation {
	protected Model model;
	protected Random rand;
	protected int MaxMCMC, burnIn, interbal, collapsedGibbsInterbal;
	private GaussianGenerator gg_proposal;

	public ArrayList<double[]> alphaR_smpl, alphaI_smpl, betaI_smpl, betaR_smpl, wT_smpl, wS_smpl, theta_smpl;
	public ArrayList<double[][]> rhoRK_smpl, topicWordDist_smpl;
	public ArrayList<double[][][]> docTopicDist_smpl;

	private int[] Nt, Zw;
	private int[][] Nwt, Nw;
	private int[][][] Ndt;
	private double[][] WriterTopicDist;
	private boolean fixedParams;

	public Estimation(Model model, Random rand, HashMap<String, Integer> settings, boolean fixedParams, String dir) {
		this.model = model; this.rand = rand; this.fixedParams = fixedParams;
		this.MaxMCMC = settings.get("MaxMCMC");
		this.burnIn = settings.get("burnIn");
		this.interbal = settings.get("interbal");
		this.collapsedGibbsInterbal = settings.get("collapsedGibbsInterbal");
		gg_proposal =  new GaussianGenerator(0.0, 0.05, rand);
		
		init();
		for(int i=0; i<this.MaxMCMC; i++){
			if( (i % (MaxMCMC/20)) == 0) {
				System.out.println("Iteration " + i +" / " + MaxMCMC + ((i >= burnIn) ? " (Sampling) " : " (Warmup) ") + "| data = " +dir + ", Given T="+ model.T);
			}
			updateThetaParam();
			if(!fixedParams) {
				updateAlphaI();
				updateAlphaR();
				updateBetaI();
				updateBetaR();
				updateRhoRK();				
			}
			if(model.T > 1) {
				if(i % collapsedGibbsInterbal == 0) {
					updateTopics();					
				}
				if(!fixedParams) {
					updateWeightT();
					updateWeightS();
				}
			}
			if(i >= burnIn && i % interbal == 0){
				this.addMCMCSamples();
			}
		}
		calc_eap();
		System.out.println(">> Estimation finished for data " +dir + ", Given T="+ model.T);
	}

	protected void init(){
		alphaI_smpl = new ArrayList<double[]>();
		alphaR_smpl = new ArrayList<double[]>();
		betaI_smpl = new ArrayList<double[]>();
		betaR_smpl = new ArrayList<double[]>();
		rhoRK_smpl = new ArrayList<double[][]>();
		theta_smpl = new ArrayList<double[]>();
		wT_smpl = new ArrayList<double[]>();
		wS_smpl = new ArrayList<double[]>();
		topicWordDist_smpl = new ArrayList<double[][]>();
		docTopicDist_smpl = new ArrayList<double[][][]>();
		
		Nwt = new int[model.V][model.T];
		Ndt = new int[model.I][model.J][model.T];
		Nw = new int[model.I][model.J];
		Nt = new int[model.T];
		Zw = new int[model.data.tokens.length];
		// assign random topic to each word
		for (int i = 0; i < Zw.length; ++i) {
			Token token = model.data.tokens[i];
			int assigned_topic = rand.nextInt(model.T);
			Nwt[token.w][assigned_topic]++;
			Ndt[token.i][token.j][assigned_topic]++;
			Nt[assigned_topic]++;
			Nw[token.i][token.j]++;
			Zw[i] = assigned_topic;
		}
		WriterTopicDist = new double[model.J][model.T];
		for (int j = 0; j < model.J; j++) {
			setWriterTopicMeanJ(j);
		}
	}
	
	void setWriterTopicMeanJ(int j) {
		model.WriterTopicMean[j] = 0;
		model.WriterTopicSigma[j] = 0;
		for (int t = 0; t < model.T; t++) {
			WriterTopicDist[j][t] = 0.0;
			for (int i = 0; i < model.I; i++) {
				WriterTopicDist[j][t] += (double) Ndt[i][j][t] / (double) Nw[i][j];
			}
			model.WriterTopicMean[j] += model.weight_t[t] * WriterTopicDist[j][t];
			model.WriterTopicSigma[j] += model.weight_s[t] * WriterTopicDist[j][t];
		}		
	}
	
	void updateWriterTopicMean() {
		for (int j = 0; j < model.J; j++) {
			model.WriterTopicMean[j] = 0;
			model.WriterTopicSigma[j] = 0;
			for (int t = 0; t < model.T; t++) {
				model.WriterTopicMean[j] += model.weight_t[t] * WriterTopicDist[j][t];
				model.WriterTopicSigma[j] += model.weight_s[t] * WriterTopicDist[j][t];
			}		
		}				
	}
	
	protected void addMCMCSamples(){
		alphaI_smpl.add(model.alpha_i.clone());
		alphaR_smpl.add(model.alpha_r.clone());
		betaI_smpl.add(model.beta_i.clone());
		betaR_smpl.add(model.beta_r.clone());
		double[][] rhoRK_tmp = new double[model.rho_rk.length][];
		for(int r=0;r<model.rho_rk.length;r++){
			rhoRK_tmp[r] = model.rho_rk[r].clone();
		}
		rhoRK_smpl.add(rhoRK_tmp);
		theta_smpl.add(model.theta.clone());
		wT_smpl.add(model.weight_t.clone());
		wS_smpl.add(model.weight_s.clone());

		double[][][] DocTopicDist = new double[model.I][model.J][model.T];
		for (int i = 0; i < model.I; i++) {
			for (int j = 0; j < model.J; j++) {
				for (int t = 0; t < model.T; t++) {
					DocTopicDist[i][j][t] = (Ndt[i][j][t] + model.alpha_lda) / (Nw[i][j] + model.alpha_lda * model.T);
				}
			}
		}
		docTopicDist_smpl.add(DocTopicDist);

		double[][] TopicWordDist = new double[model.T][model.V];
		for (int i = 0; i < model.T; ++i) {
			double sum = 0.0;
			for (int j = 0; j < model.V; ++j) {
				TopicWordDist[i][j] = model.beta_lda + Nwt[j][i];
				sum += TopicWordDist[i][j];
			}
			for (int j = 0; j < model.V; ++j) {
				TopicWordDist[i][j] /= sum;
			}
		}
		topicWordDist_smpl.add(TopicWordDist);
	}
	
	public void updateAlphaI(){
		double[] prevAlphaI = model.alpha_i.clone();
		double sum = 0.0;
		double prevLikelihood = model.LogLikelihoodIRT();
		for(int i=0;i<model.I;i++){
			prevLikelihood += Math.log(MyUtil.gaussian_value(model.alpha_prior[0], model.alpha_prior[1], model.alpha_i[i]));
			model.alpha_i[i] += gg_proposal.nextValue();
			sum += model.alpha_i[i];
		}
		sum = sum / model.I;
		double newLikelihood = 0.0;
		for(int i=0;i<model.I;i++){
			model.alpha_i[i] = model.alpha_i[i] - sum;
			newLikelihood += Math.log(MyUtil.gaussian_value(model.alpha_prior[0], model.alpha_prior[1], model.alpha_i[i]));
		}
		newLikelihood += model.LogLikelihoodIRT();
		double thresh = Math.exp(newLikelihood - prevLikelihood);
		if(MyUtil.isRejected(thresh, rand) == true){
			model.alpha_i = prevAlphaI.clone();
		}
	}

	public void updateAlphaR(){
		for(int r=0;r<model.R;r++){
			double prevAlphaR = model.alpha_r[r];
			double prevLikelihood = model.LogLikelihoodRater(r) + Math.log(MyUtil.gaussian_value(model.alpha_prior[0], model.alpha_prior[1], model.alpha_r[r]));
			model.alpha_r[r] += gg_proposal.nextValue();
			double newLikelihood = model.LogLikelihoodRater(r) + Math.log(MyUtil.gaussian_value(model.alpha_prior[0], model.alpha_prior[1], model.alpha_r[r]));
			double thresh = Math.exp(newLikelihood - prevLikelihood);
			if(MyUtil.isRejected(thresh, rand) == true){
				model.alpha_r[r] = prevAlphaR;
			}
		}
	}

	public void updateBetaI(){
		double[] prevBetaI = model.beta_i.clone();
		double sum = 0.0;
		double prevLikelihood = model.LogLikelihoodIRT();
		for(int i=0;i<model.I;i++){
			prevLikelihood += Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1], model.beta_i[i]));
			model.beta_i[i] += gg_proposal.nextValue();
			sum += model.beta_i[i];
		}
		sum = sum / model.I;
		double newLikelihood = 0.0;
		for(int i=0;i<model.I;i++){
			model.beta_i[i] = model.beta_i[i] - sum;
			newLikelihood += Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1], model.beta_i[i]));
		}
		newLikelihood += model.LogLikelihoodIRT();
		double thresh = Math.exp(newLikelihood - prevLikelihood);
		if(MyUtil.isRejected(thresh, rand) == true){
			model.beta_i = prevBetaI.clone();
		}
	}

	protected void updateBetaR(){
		for(int r=0;r<model.R;r++){
			double prevBetaR = model.beta_r[r];
			double prevLikelihood = model.LogLikelihoodRater(r) + Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1], model.beta_r[r]));
			model.beta_r[r] += gg_proposal.nextValue();
			double newLikelihood = model.LogLikelihoodRater(r) + Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1], model.beta_r[r]));
			double thresh = Math.exp(newLikelihood - prevLikelihood);
			if(MyUtil.isRejected(thresh, rand) == true){
				model.beta_r[r] = prevBetaR;
			}
		}
	}

	public void updateRhoRK(){
		for(int r=0;r<model.R;r++){
			if(model.K > 2){
				double[] prevRhoRK = model.rho_rk[r].clone();
				double sum = 0.0;
				double prevLikelihood = model.LogLikelihoodRater(r);
				for(int k=1;k<model.K;k++){
					prevLikelihood += Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1],model.rho_rk[r][k]));
					model.rho_rk[r][k] += gg_proposal.nextValue();
					sum += model.rho_rk[r][k];
				}
				sum = sum / (model.K - 1.0);
				double newLikelihood = 0.0;
				for(int k=1;k<model.K;k++){
					model.rho_rk[r][k] = model.rho_rk[r][k] - sum;
					newLikelihood += Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1],model.rho_rk[r][k]));
				}
				newLikelihood += model.LogLikelihoodRater(r);
				double thresh = Math.exp(newLikelihood - prevLikelihood);
				if( MyUtil.isRejected(thresh, rand) == true){
					model.rho_rk[r] = prevRhoRK.clone();
				}
				prevRhoRK = null;
			}
		}
	}

	public void updateThetaParam(){
		for(int j=0;j<model.J;j++){
			double prevLikelihood = model.LogLikelihoodExaminee(j) + model.LogProbabilityThetaJ(j);
			double prevTheta = model.theta[j];
			model.theta[j] += gg_proposal.nextValue();
			double newLikelihood = model.LogLikelihoodExaminee(j) + model.LogProbabilityThetaJ(j);
			double thresh = Math.exp(newLikelihood - prevLikelihood);
			if( MyUtil.isRejected(thresh, rand) == true){
				model.theta[j] = prevTheta;
			}
		}
	}
	
	public void updateWeightT(){
		double[] prevWt = model.weight_t.clone();
		double prevLikelihood = model.LogProbabilityTheta();
		double newLikelihood = 0;
		for(int t=0;t<model.T;t++){
			prevLikelihood += Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1], model.weight_t[t]));
			model.weight_t[t] += gg_proposal.nextValue();
			newLikelihood += Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1], model.weight_t[t]));
		}
		updateWriterTopicMean();
		newLikelihood += model.LogProbabilityTheta();
		double thresh = Math.exp(newLikelihood - prevLikelihood);
		if(MyUtil.isRejected(thresh, rand) == true){
			model.weight_t = prevWt.clone();
			updateWriterTopicMean();
		}
	}

	public void updateWeightS(){
		double[] prevWs = model.weight_s.clone();
		double prevLikelihood = model.LogProbabilityTheta();
		double newLikelihood = 0;
		for(int t=0;t<model.T;t++){
			prevLikelihood += Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1], model.weight_s[t]));
			model.weight_s[t] += gg_proposal.nextValue();
			newLikelihood += Math.log(MyUtil.gaussian_value(model.normal_prior[0], model.normal_prior[1], model.weight_s[t]));
		}
		updateWriterTopicMean();
		newLikelihood += model.LogProbabilityTheta();
		double thresh = Math.exp(newLikelihood - prevLikelihood);
		if(MyUtil.isRejected(thresh, rand) == true){
			model.weight_s = prevWs.clone();
			updateWriterTopicMean();
		}
	}	

	private void updateTopics() {
		for (int tId = 0; tId < Zw.length; ++tId) {
			Token token = model.data.tokens[tId];
			int assign = Zw[tId];
			Nwt[token.w][assign]--;
			Ndt[token.i][token.j][assign]--;
			Nt[assign]--;

			// calculate posterior cumulative multi-nominal distribution
			double[] P = new double[model.T];
			for (int t = 0; t < P.length; ++t) {
				if(fixedParams) {
					P[t] = model.TopicWordDist[t][token.w];
				} else {
					P[t] = ((double)Nwt[token.w][t] + model.beta_lda);					
				}
				P[t] *= ((double)Ndt[token.i][token.j][t] + model.alpha_lda) / ((double)Nt[t] + model.T * model.alpha_lda);
				Ndt[token.i][token.j][t]++;
				setWriterTopicMeanJ(token.j);
				P[t] *= Math.exp(model.LogProbabilityThetaJ(token.j));
				Ndt[token.i][token.j][t]--;
				if (t != 0) P[t] += P[t - 1];
			}
			
			// random selection
			double u = rand.nextDouble() * P[model.T - 1];
			assign = model.T - 1;
			for (int t = 0; t < P.length; ++t) {
				if (u < P[t]) {
					assign = t;
					break;
				}
			}
			
			// update count
			Ndt[token.i][token.j][assign]++;
			Nt[assign]++;
			Nwt[token.w][assign]++;
			Zw[tId] = assign;
			setWriterTopicMeanJ(token.j);
		}
	}	
	
	protected void calc_eap(){
		model.alpha_i = MyUtil.get_averages(alphaI_smpl, 0, 1);
		model.alpha_r = MyUtil.get_averages(alphaR_smpl, 0, 1);
		model.beta_i = MyUtil.get_averages(betaI_smpl, 0, 1);
		model.beta_r = MyUtil.get_averages(betaR_smpl, 0, 1);
		model.rho_rk = MyUtil.get_averages_matrix(rhoRK_smpl, 0, 1);
		model.theta = MyUtil.get_averages(theta_smpl, 0, 1);
		model.weight_t = MyUtil.get_averages(wT_smpl, 0, 1);
		model.weight_s = MyUtil.get_averages(wS_smpl, 0, 1);
		model.TopicWordDist = MyUtil.get_averages_matrix(topicWordDist_smpl, 0, 1);
		model.DocTopicDist = MyUtil.get_averages_3Dmatrix(docTopicDist_smpl, 0, 1);
		for (int i = 0; i < model.T; ++i) {
			model.TopicWordDist[i] = MyUtil.toProb(model.TopicWordDist[i]);
		}
		for (int i = 0; i <model. I; i++) {
			for (int j = 0; j < model.J; j++) {
				model.DocTopicDist[i][j] = MyUtil.toProb(model.DocTopicDist[i][j]);
			}
		}
	}
}
