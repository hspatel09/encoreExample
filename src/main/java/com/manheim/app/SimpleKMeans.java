package com.manheim.app;



import java.util.Arrays;

import org.encog.ml.MLCluster;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.kmeans.KMeansClustering;

/**
 * This example performs a simple KMeans cluster.  The numbers are clustered
 * into two groups.
 */
public class SimpleKMeans {

	/**
	 * The data to be clustered.
	 */
	public static final double[][] DATA = { { 28, 15, 22 }, { 16, 15, 32 },
			{ 32, 20, 44 }, { 1, 2, 3 },{2 , 3, 1 }, { 3, 2, 1 } ,{28, 15, 22 },{28.1, 15.4, 22.3 }};

	public static final double[][] DATA2 = { { 5,7,11}, { 4,6, 8},
			{ 11, 22, 33 },{17, 19, 23}, { 3, 9, 12 },{44,55,66 }};
	
	public static final double[][] DATASET =  { {5,3}, {10,15}, {15,12}, {24,10}, {30,45}, {85,70}, {71,80}, {60,78}, {55,52}, {80,91} } ; 
	/**
	 * The main method.
	 * @param args Arguments are not used.
	 */
	public static void main(final String args[]) {

		final BasicMLDataSet set = new BasicMLDataSet();
	
		int o = 0 ;
		for (final double[] element : SimpleKMeans.DATASET) {
		//	System.out.println(element[0]);
		//	System.out.println(element[1]);
		//	System.out.println(element[2]);
		//	System.out.println("***");
			set.add(new BasicMLData(element));
		}

		final KMeansClustering kmeans = new KMeansClustering(4, set);
		
		kmeans.iteration(4);
	
		// Display the cluster
		int i = 1;
		for (final MLCluster cluster : kmeans.getClusters()) {
			System.out.println("*** Cluster " + (i++) + " ***");
			final MLDataSet ds = cluster.createDataSet();
			final MLDataPair pair = BasicMLDataPair.createPair(
					ds.getInputSize(), ds.getIdealSize());
			for (int j = 0; j < ds.getRecordCount(); j++) {
				ds.getRecord(j, pair);
				System.out.println(Arrays.toString(pair.getInputArray()));

			}
		}
	}
}
