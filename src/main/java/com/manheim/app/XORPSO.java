package com.manheim.app;


import org.encog.Encog;
import org.encog.mathutil.randomize.NguyenWidrowRandomizer;
import org.encog.mathutil.randomize.Randomizer;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.pso.NeuralPSO;
import org.encog.util.simple.EncogUtility;

/**
 * XOR-PSO: This example solves the classic XOR operator neural
 * network problem.  However, it uses PSO training.
 * 
 * @author $Author$
 * @version $Revision$
 */
public class XORPSO {
	public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
			{ 0.0, 1.0 }, { 1.0, 1.0 } };

	public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

	public static void main(final String args[]) {

		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		BasicNetwork network = EncogUtility.simpleFeedForward(2, 2, 0, 1, false);
		CalculateScore score = new TrainingSetScore(trainingSet);
		Randomizer randomizer = new NguyenWidrowRandomizer();
		
		final MLTrain train = new NeuralPSO(network,randomizer,score,20);
		
		EncogUtility.trainToError(train, 0.01);

		network = (BasicNetwork)train.getMethod();

		// test the neural network
		System.out.println("Neural Network Results:");
		EncogUtility.evaluate(network, trainingSet);
		
		Encog.getInstance().shutdown();
	}
}
