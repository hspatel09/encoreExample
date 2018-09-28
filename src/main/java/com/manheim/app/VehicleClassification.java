package com.manheim.app;


import java.io.File;
import java.net.MalformedURLException;
import java.util.Arrays;

import org.encog.ConsoleStatusReportable;
import org.encog.Encog;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

public class VehicleClassification {
//	public static String DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";

	private String tempPath;

	public File downloadData(String[] args) throws MalformedURLException {
		if (args.length != 0) {
			tempPath = args[0];
		} else {
			tempPath = System.getProperty("java.io.tmpdir");
			System.out.println("--------------");
			System.out.println(tempPath);
			System.out.println("--------------");
		}

		ClassLoader classLoader = getClass().getClassLoader();
		File vehicleFile = new File(classLoader.getResource("vehicle.csv").getFile());
	//	BotUtil.downloadPage(new URL(IrisClassification.DATA_URL), irisFile);
		System.out.println("Downloading vehicle dataset to: " + vehicleFile);
		return vehicleFile;
	}

	public void run(String[] args) {
		try {
			// Download the data that we will attempt to model.
			File vehicleFile = downloadData(args);
			
			// Define the format of the data file.
			// This area will change, depending on the columns and 
			// format of the file that you are trying to model.
			
			VersatileDataSource source = new CSVDataSource(vehicleFile, false,
					CSVFormat.ENGLISH);
			VersatileMLDataSet data = new VersatileMLDataSet(source);
		
			data.defineSourceColumn("model", 1, ColumnType.nominal);
			data.defineSourceColumn("trim", 2, ColumnType.nominal);
			
			// Define the column that we are trying to predict.
			ColumnDefinition outputColumn = data.defineSourceColumn("make", 0,
					ColumnType.nominal);
			
			// Analyze the data, determine the min/max/mean/sd of every column.
			data.analyze();
			
			// Map the prediction column to the output of the model, and all
			// other columns to the input.
			data.defineSingleOutputOthersInput(outputColumn);
			
			// Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
			// You could also other model types, such as:
			// MLMethodFactory.SVM:  Support Vector Machine (SVM)
			// MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
			// MLMethodFactor.TYPE_NEAT: NEAT Neural Network
			// MLMethodFactor.TYPE_PNN: Probabilistic Neural Network
			EncogModel model = new EncogModel(data);
			//model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
			model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
			// Send any output to the console.
			model.setReport(new ConsoleStatusReportable());
			
			// Now normalize the data.  Encog will automatically determine the correct normalization
			// type based on the model you chose in the last step.
			data.normalize();
			
			// Hold back some data for a final validation.
			// Shuffle the data into a random ordering.
			// Use a seed of 1001 so that we always use the same holdback and will get more consistent results.
			model.holdBackValidation(0.3, true, 1001);
			
			// Choose whatever is the default training type for this model.
			model.selectTrainingType(data);
			
			// Use a 5-fold cross-validated train.  Return the best method found.
			MLRegression bestMethod = (MLRegression)model.crossvalidate(5, true);

			System.out.println( "Configs features: " + 	model.getMethodConfigurations());
		
			// Display the training and validation errors.
			System.out.println( "Training error: " + EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
			System.out.println( "Validation error: " + EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));
			
			// Display our normalization parameters.
			NormalizationHelper helper = data.getNormHelper();
			System.out.println("HELPER:");
			System.out.println(helper.toString());
			
			// Display the final model.
			System.out.println("Final model: " + bestMethod);
			
			// Loop over the entire, original, dataset and feed it through the model.
			// This also shows how you would process new data, that was not part of your
			// training set.  You do not need to retrain, simply use the NormalizationHelper
			// class.  After you train, you can save the NormalizationHelper to later
			// normalize and denormalize your data.
			ReadCSV csv = new ReadCSV(vehicleFile, false, CSVFormat.ENGLISH);
			System.out.println(csv);
			String[] line = new String[2];
			MLData input = helper.allocateInputVector();
			
			int errorCounter = 0;
			int iterationCount = 0;
			while(csv.next()) {
				StringBuilder result = new StringBuilder();
				line[0] = csv.get(1);
				line[1] = csv.get(2);
				System.out.println("line[0]:" +line[0]); 
						System.out.println("line[1]:" + line[1]);
				String correct = csv.get(0);
				
				
				helper.normalizeInputVector(line,input.getData(),true);
				
				
				MLData output = bestMethod.compute(input);
				System.out.println("MLData:" + output.getData()[0]);
				String irisChosen = helper.denormalizeOutputVectorToString(output)[0];
				
				result.append(Arrays.toString(line));
				result.append(" -> predicted: ");
				result.append(irisChosen);
				result.append("(correct: ");
				result.append(correct);
				result.append(")");
				if(!irisChosen.equalsIgnoreCase(correct)) {
					errorCounter++;
					System.err.println(result.toString());
				}{
					System.out.println(result.toString());
				}
				iterationCount++;
				
			}
			System.out.println("Out of " + iterationCount + " results " + errorCounter + " are incorrect");
			// Delete data file ande shut down.
			vehicleFile.delete();
			Encog.getInstance().shutdown();

		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	public static void main(String[] args) {
		VehicleClassification prg = new VehicleClassification();
		prg.run(args);
	}
}
