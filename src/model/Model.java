package model;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Random;

import utility.Data;
import utility.MyUtil;

public class Model {
	public int I, J, R, T, V, K;
	public double[] alpha_r, alpha_i, beta_i, beta_r, theta, weight_t, weight_s;
	public double[][] rho_rk, TopicWordDist;
	public double[][][] DocTopicDist;
	public double[] WriterTopicMean;
	public double[] WriterTopicSigma;

	public double[] alpha_prior = { 0.0, 1.0 };
	public double[] normal_prior = { 0.0, 1.0 };
	double alpha_lda, beta_lda;

	protected Random rand;
	public Data data;

	public Model(int I, int J, int R, int T, int V, int K, Random rand){
		this.I = I; this.J = J; this.R = R;
		this.T = T; this.V = V; this.K = K;
		this.rand = rand;

		// initialize parameters
		alpha_i = new double[I];
		alpha_r = new double[R];
		beta_i = new double[I];
		beta_r = new double[R];
		rho_rk = new double[R][K];
		theta = new double[J];
		weight_t = new double[T];
		weight_s = new double[T];
		if(T == 1) {
			weight_t[0] = 0; 
			weight_s[0] = 1; 
		}
		DocTopicDist = new double[I][J][T];
		TopicWordDist = new double[T][V];
		WriterTopicMean = new double[J];
		WriterTopicSigma = new double[J];

		// init data
		this.data = new Data(I, J, R, rand);
		
		// set hyper parameters
		this.alpha_lda = 1.0 / T;
		this.beta_lda = 1.0/(T*V);
	}

	public double[] IRC(int i, int j, int r) {
		double[] IRC = new double[K];
		double[] Zij = new double[K];
		double Z = 0;
		double Z_tmp = 0;
		for (int k = 0; k < K; k++) {
			Z_tmp += Math.exp(alpha_i[i]) * Math.exp(alpha_r[r]) * (theta[j] - beta_i[i] - beta_r[r] - rho_rk[r][k]);
			Zij[k] = Math.exp(Z_tmp);
			Z += Zij[k];
		}
		for (int k = 0; k < K; k++) {
			IRC[k] += (Zij[k] / Z);
		}
		return IRC;
	}

	public double getProbability(int i, int j, int r) {
		int k = this.data.U[i][j][r];
		if (k != -1) {
			return IRC(i, j, r)[k];
		}
		return 1.0;
	}

	double LogLikelihoodIRT() {
		double LogLikelihood = 0.0;
		for (int j = 0; j < J; j++) {
			LogLikelihood += LogLikelihoodExaminee(j);
		}
		return LogLikelihood;
	}

	double LogLikelihoodExaminee(int j) {
		double LogLikelihood = 0.0;
		for (int i = 0; i < I; i++) {
			for (int r = 0; r < R; r++) {
				LogLikelihood += Math.log(getProbability(i, j, r));
			}
		}
		return LogLikelihood;
	}

	double LogLikelihoodRater(int r) {
		double LogLikelihood = 0.0;
		for (int i = 0; i < I; i++) {
			for (int j = 0; j < J; j++) {
				LogLikelihood += Math.log(getProbability(i, j, r));
			}
		}
		return LogLikelihood;
	}

	double LogLikelihoodItem(int i) {
		double LogLikelihood = 0.0;
		for (int j = 0; j < J; j++) {
			for (int r = 0; r < R; r++) {
				LogLikelihood += Math.log(getProbability(i, j, r));
			}
		}
		return LogLikelihood;
	}

	double LogProbabilityTheta() {
		double probTheta = 0;
		for (int j = 0; j < J; j++) {
			probTheta += LogProbabilityThetaJ(j);
		}
		return probTheta;
	}

	double LogProbabilityThetaJ(int j) {
		return Math.log(MyUtil.gaussian_value(WriterTopicMean[j], Math.exp(WriterTopicSigma[j]), theta[j]));
	}

	public void set_parameters(String filename) throws IOException {
		BufferedReader br = MyUtil.Reader(filename);
		br.readLine(); // information criteria
		String[] alpha_i = br.readLine().split(",");
		String[] beta_i = br.readLine().split(",");
		String[] alpha_r = br.readLine().split(",");
		String[] beta_r = br.readLine().split(",");
		String[][] tau_rk = new String[K][];
		for(int k=0;k<K;k++) {
			tau_rk[k] = br.readLine().split(",");
		}
		String[] theta = br.readLine().split(",");
		String[][] wotdDist = new String[T][];
		for(int t=0;t<T;t++) {
			wotdDist[t] = br.readLine().split(",");
		}
		String[] weight_mu = br.readLine().split(",");
		String[] weight_sigma = br.readLine().split(",");
		String[][][] topicDist = new String[J][I][];
		for(int j=0;j<J;j++) {
			for(int i=0;i<I;i++) {
				topicDist[j][i] = br.readLine().split(",");
			}
		}
		for(int i=0;i<I;i++) {
			this.alpha_i[i] = Double.valueOf(alpha_i[i+1]);
			this.beta_i[i] = Double.valueOf(beta_i[i+1]);
		}
		for(int r=0;r<R;r++) {
			this.alpha_r[r] = Double.valueOf(alpha_r[r+1]);
			this.beta_r[r] = Double.valueOf(beta_r[r+1]);
			for(int k=0;k<K;k++) {
				this.rho_rk[r][k] = Double.valueOf(tau_rk[k][r+1]);
			}
		}
//		for(int j=0;j<J;j++) {
//			this.theta[j] = Double.valueOf(theta[j+1]);
//		}
		for(int t=0;t<T;t++) {
			for(int v=0;v<V;v++) {
				this.TopicWordDist[t][v] = Double.valueOf(wotdDist[t][v+1]);
			}
		}		
		for(int t=0;t<T;t++) {
			this.weight_t[t] = Double.valueOf(weight_mu[t+1]);
			this.weight_s[t] = Double.valueOf(weight_sigma[t+1]);
		}
//		for(int j=0;j<J;j++) {
//			for(int i=0;i<I;i++) {
//				for(int t=0;t<T;t++) {
//					this.DocTopicDist[i][j][t] = Double.valueOf(topicDist[j][i][t+1]);
//				}
//			}
//		}
//		for (int j = 0; j < J; j++) {
//			WriterTopicMean[j] = 0;
//			WriterTopicSigma[j] = 0;
//			for (int t = 0; t < T; t++) {
//				double[] WriterTopicDist = new double[T];
//				for (int i = 0; i < I; i++) {
//					WriterTopicDist[t] += DocTopicDist[i][j][t];
//				}
//				WriterTopicMean[j] += weight_t[t] * WriterTopicDist[t];
//				WriterTopicSigma[j] += weight_s[t] * WriterTopicDist[t];
//			}
//			if(T == 1) WriterTopicSigma[j] = 1;
//		}		
	}
	
	public void set_true_parameters(String filename) throws IOException {
		BufferedReader br = MyUtil.Reader(filename);
		String[] alpha_i = br.readLine().split(",");
		String[] beta_i = br.readLine().split(",");
		String[] alpha_r = br.readLine().split(",");
		String[] beta_r = br.readLine().split(",");
		String[][] tau_rk = new String[K][];
		for(int k=0;k<K;k++) {
			tau_rk[k] = br.readLine().split(",");
		}
		String[] theta = br.readLine().split(",");
		String[][] wotdDist = new String[T][];
		for(int t=0;t<T;t++) {
			wotdDist[t] = br.readLine().split(",");
		}
		String[] weight_mu = br.readLine().split(",");
		String[] weight_sigma = br.readLine().split(",");
		String[][][] topicDist = new String[J][I][];
		for(int j=0;j<J;j++) {
			for(int i=0;i<I;i++) {
				topicDist[j][i] = br.readLine().split(",");
			}
		}
		for(int i=0;i<I;i++) {
			this.alpha_i[i] = Double.valueOf(alpha_i[i]);
			this.beta_i[i] = Double.valueOf(beta_i[i]);
		}
		for(int r=0;r<R;r++) {
			this.alpha_r[r] = Double.valueOf(alpha_r[r]);
			this.beta_r[r] = Double.valueOf(beta_r[r]);
			for(int k=0;k<K;k++) {
				this.rho_rk[r][k] = Double.valueOf(tau_rk[k][r]);
			}
		}
		for(int t=0;t<T;t++) {
			for(int v=0;v<V;v++) {
				this.TopicWordDist[t][v] = Double.valueOf(wotdDist[t][v]);
			}
		}		
		for(int t=0;t<T;t++) {
			this.weight_t[t] = Double.valueOf(weight_mu[t]);
			this.weight_s[t] = Double.valueOf(weight_sigma[t]);
		}
	}	

	public void setEapTheta(){
		int pointNum = 50;
		for(int j=0;j<J;j++){
			double[] Xh = new double[pointNum];
			double[] AXh = new double[pointNum];
			Xh[0] = WriterTopicMean[j] - 4 *Math.exp(WriterTopicSigma[j]);
			double delta = (8 * Math.exp(WriterTopicSigma[j])) / (pointNum -1);
			AXh[0] = MyUtil.gaussian_value(WriterTopicMean[j], Math.exp(WriterTopicSigma[j]), Xh[0]);
			for(int i=1;i<pointNum;i++){
				Xh[i] = Xh[i-1] +delta;
				AXh[i] = MyUtil.gaussian_value(WriterTopicMean[j], Math.exp(WriterTopicSigma[j]), Xh[i]);
			}

			double eq1 = 0.0;
			double eq2 = 0.0;
			for(int h=0;h<pointNum;h++){
				double LL = 0.0;
				for(int r=0;r<R;r++){
					for(int i=0;i<I;i++){
						int u = this.data.U[i][j][r];
						if(u!=-1){
							double[] IRC = new double[this.K];
							double[] Zij = new double[this.K];
							double Z = 0;
							double Z_tmp = 0;
							for (int k = 0; k < this.K; k++) {
								Z_tmp += Math.exp(alpha_i[i]) * Math.exp(alpha_r[r]) * (Xh[h] - beta_i[i] - beta_r[r] - rho_rk[r][k]);
								Zij[k] = Math.exp(Z_tmp);
								Z += Zij[k];
							}
							for (int k = 0; k < this.K; k++) {
								IRC[k] += (Zij[k] / Z);
							}							
							LL += Math.log(IRC[u]);
						}
					}
				}
				eq1 += Math.exp(LL)*AXh[h]*Xh[h];
				eq2 += Math.exp(LL)*AXh[h];
			}
			theta[j] = eq1 / eq2;
		}
	}
}
