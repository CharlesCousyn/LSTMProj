import GENERAL_CONFIG from "./configFiles/generalConfig";
import filesSystem from "fs"
import csvToJson from "csv-file-to-json"
import * as DATASET_INIT from "./datasetInit";
import * as tensorflow from "@tensorflow/tfjs-node-gpu"
import {jsonDocumentsToTensors, myTokenizer} from "./dataUtils";

function buildModel(maxLen, vocabularySize, embeddingSize, numClasses)
{
	const model = tensorflow.sequential();

	model.add(tensorflow.layers.embedding(
		{
			inputDim: vocabularySize,
			outputDim: embeddingSize,//embeddingSize = 32
			inputLength: maxLen//maxLen = 250
		}));

	model.add(tensorflow.layers.lstm({units: maxLen, returnSequences: false, recurrentInitializer: 'glorotNormal'}));

	/*const dense = tensorflow.layers.dense({units: numClasses});
	model.add(tensorflow.layers.timeDistributed({layer: dense}));
	model.add(tensorflow.layers.activation({activation: 'softmax'}));*/
	model.add(tensorflow.layers.dense({units: numClasses, activation: 'softmax'}));//numClasses = 3

	model.summary();
	return model;
}

async function trainModel(model, data, labels, epochs, batchSize, validationSplit, modelSaveDir)
{
	console.log('Training model...');
	const history = await model.fit(data, labels, {
		epochs: epochs,
		batchSize: batchSize,
		validationSplit: validationSplit,
		callbacks: () =>
		{
			console.log("Coucou");
		}
	});
	console.log(history);

	console.log('Evaluating model...');
	const [testLoss, testAcc] = model.evaluate(data, labels, {batchSize: 100});
	console.log(`Evaluation loss: ${(await testLoss.data())[0].toFixed(4)}`);
	console.log(`Evaluation accuracy: ${(await testAcc.data())[0].toFixed(4)}`);

	// Save model artifacts.
	await model.save(`file://${modelSaveDir}`);
	console.log(`Saved model to ${modelSaveDir}`);
}

(async () =>
{
	const objectWithLabels = csvToJson({ filePath: GENERAL_CONFIG.pathToLabelFile });
	const documents = DATASET_INIT.filesDocumentToJsonDocument(
		GENERAL_CONFIG.pathToDataFileDocuments,
		GENERAL_CONFIG.pathToFolderOfTextFiles,
		objectWithLabels,
		GENERAL_CONFIG.labelMapping);

	//const {dataTensorsTrain, dataTensorsTest,  labelTensorsTrain, labelTensorsTest} = jsonDocumentsToTensors(documents, GENERAL_CONFIG.sequenceSizePerDocument);
	const {dataTensors, labelTensors} = jsonDocumentsToTensors(documents, GENERAL_CONFIG.sequenceSizePerDocument);


	const model = buildModel(
		GENERAL_CONFIG.sequenceSizePerDocument,
		myTokenizer.wordCounts.length,
		GENERAL_CONFIG.embeddingSize,
		Object.keys(GENERAL_CONFIG.labelMapping).length);

	model.compile({
		loss: 'categoricalCrossentropy',
		optimizer: tensorflow.train.rmsprop(GENERAL_CONFIG.learningRate),
		metrics: ["accuracy"]
	});

	await trainModel(model, dataTensors, labelTensors,
		GENERAL_CONFIG.epochs,
		GENERAL_CONFIG.batchSize,
		GENERAL_CONFIG.validationSplit,
		GENERAL_CONFIG.modelSaveDir);
})();