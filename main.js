import GENERAL_CONFIG from "./configFiles/generalConfig";
import filesSystem from "fs"
import csvToJson from "csv-file-to-json"
import * as DATASET_INIT from "./datasetInit";
import * as tensorflow from "@tensorflow/tfjs-node-gpu"
import {jsonDocumentsToTensors, myTokenizer} from "./dataUtils";

function buildModel(maxLen, vocabularySize, embeddingSize, numClasses, seed, numberLSTMCells)
{
	console.log("maxLen", maxLen);
	console.log("vocabularySize", vocabularySize);
	console.log("embeddingSize", embeddingSize);
	console.log("numClasses", numClasses);
	const model = tensorflow.sequential();

	const embeddingLayer = tensorflow.layers.embedding(
		{
			inputDim: vocabularySize,
			outputDim: embeddingSize,
			inputLength: maxLen,
			embeddingsInitializer: tensorflow.initializers.glorotNormal(seed)
		});

	const lstmLayer = tensorflow.layers.bidirectional({layer: tensorflow.layers.lstm({units: numberLSTMCells, returnSequences: true, recurrentInitializer: 'glorotNormal', unitForgetBias: true}), mergeMode: 'concat'});
	const lstmLayer2 = tensorflow.layers.bidirectional({layer: tensorflow.layers.lstm({units: numberLSTMCells, returnSequences: false, recurrentInitializer: 'glorotNormal', unitForgetBias: true}), mergeMode: 'concat'});
	const outputLayer = tensorflow.layers.dense(
		{
			units: numClasses,
			activation: 'softmax',
			kernelInitializer: tensorflow.initializers.glorotNormal(seed),
			biasInitializer: tensorflow.initializers.glorotNormal(seed),
			useBias: true
		});

	model.add(embeddingLayer);
	model.add(lstmLayer);
	model.add(lstmLayer2);
	model.add(outputLayer);

	model.summary();
	return model;
}

async function trainModel(model, dataTrain, labelsTrain, dataTest, labelsTest, epochs, batchSize, classWeight)
{
	console.log('Training model...');
	const history = await model.fit(dataTrain, labelsTrain, {
		epochs: epochs,
		batchSize: batchSize,
		validationData:[dataTest, labelsTest],
		callbacks:
			{
				onEpochEnd: async(epoch, logs) =>
				{
					const dir = `./modelSave${epoch}`;
					if (!filesSystem.existsSync(dir))
					{
						filesSystem.mkdirSync(dir);
					}

					await saveModel(model, dir);
				}
			},
		classWeight: classWeight
	});
	console.log(history);

	//Free Memory
	tensorflow.dispose([dataTrain, labelsTrain]);

	console.log('Evaluating model...');
	const [testLoss, testAcc] = model.evaluate(dataTest, labelsTest, {batchSize: 100});

	//Free Memory
	tensorflow.dispose([dataTest, labelsTest]);

	console.log(`Evaluation loss: ${(await testLoss.data())[0].toFixed(4)}`);
	console.log(`Evaluation accuracy: ${(await testAcc.data())[0].toFixed(4)}`);
}

async function saveModel(model, modelSaveDir)
{
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
	const {dataTensorsTrain, labelTensorsTrain, dataTensorsTest, labelTensorsTest} = jsonDocumentsToTensors(documents, GENERAL_CONFIG.sequenceSizePerDocument, GENERAL_CONFIG.validationSplit);


	const model = buildModel(
		GENERAL_CONFIG.sequenceSizePerDocument,
		Object.keys(myTokenizer.wordCounts).length,
		GENERAL_CONFIG.embeddingSize,
		Object.keys(GENERAL_CONFIG.labelMapping).length,
		GENERAL_CONFIG.seed,
		GENERAL_CONFIG.numberLSTMCells);

	model.compile({
		loss: 'categoricalCrossentropy',
		optimizer: tensorflow.train.adam(GENERAL_CONFIG.learningRate),
		metrics: ["accuracy"]
	});

	await trainModel(model, dataTensorsTrain, labelTensorsTrain, dataTensorsTest, labelTensorsTest,
		GENERAL_CONFIG.epochs,
		GENERAL_CONFIG.batchSize,
		GENERAL_CONFIG.classWeight);

	await saveModel(model, GENERAL_CONFIG.modelSaveDir);
})();