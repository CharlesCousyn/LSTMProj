import tokenizer from "./tokenizer.js"
import * as tensorflow from "@tensorflow/tfjs-node-gpu"
import readline from 'readline'

export let myTokenizer = null;

function initTokenizer(documents)
{
	myTokenizer = new tokenizer();
	const complaintsTexts = documents.map(doc => doc.text);
	myTokenizer.fitOnTexts(complaintsTexts);
}

export function jsonDocumentsToTensors(documents, sequenceSizePerDocument, validationSplit)
{
	initTokenizer(documents);

	let wordVectorsOriginal = myTokenizer.textsToSequences(documents.map(doc => doc.text));
	let labels = documents.map(doc => doc.labelNN);
	//labels = [labels[0], labels[1], labels[13]];

	console.log(getInstancesPerLabel(labels));

	//Keeping maxWordsPerDocument elements per document and adding 0 when too short
	wordVectorsOriginal = padSequences(wordVectorsOriginal, sequenceSizePerDocument, "post", "pre", 0);
	//wordVectorsOriginal = [wordVectorsOriginal[0], wordVectorsOriginal[1], wordVectorsOriginal[13]];

	//Shuffle data and labels
	[wordVectorsOriginal, labels] = shuffleTwoArrays(wordVectorsOriginal, labels);

	//Split train and test dataset
	const indexOfSplit = Math.round(validationSplit * wordVectorsOriginal.length);
	const wordVectorsOriginalTrain = wordVectorsOriginal.slice(0, indexOfSplit);
	const wordVectorsOriginalTest = wordVectorsOriginal.slice(indexOfSplit, wordVectorsOriginal.length);
	const labelsTrain = labels.slice(0, indexOfSplit);
	const labelsTest = labels.slice(indexOfSplit, labels.length);

	//Create tensors
	const dataTensorsTrain = tensorflow.tensor2d(wordVectorsOriginalTrain);
	const dataTensorsTest = tensorflow.tensor2d(wordVectorsOriginalTest);
	const labelTensorsTrain = tensorflow.tensor2d(labelsTrain);
	const labelTensorsTest = tensorflow.tensor2d(labelsTest);
	console.log("dataTensorsTrain.shape", dataTensorsTrain.shape, "labelTensorsTrain.shape", labelTensorsTrain.shape, "dataTensorsTest.shape", dataTensorsTest.shape, "labelTensorsTest.shape", labelTensorsTest.shape);
	return {dataTensorsTrain, labelTensorsTrain, dataTensorsTest, labelTensorsTest};
}

function shuffleTwoArrays(array1, array2)
{
	for(let i = array1.length - 1; i > 0; i--)
	{
		const j = Math.floor(Math.random() * i);

		const temp1 = array1[i];
		array1[i] = array1[j];
		array1[j] = temp1;

		const temp2 = array2[i];
		array2[i] = array2[j];
		array2[j] = temp2;
	}
	return [array1, array2];
}

function getStats(wordVectorsOriginal, tokenizer)
{
	return {
		numberOfInstances: wordVectorsOriginal.length,
		maxSizeInstance: Math.max(...wordVectorsOriginal.map(vec => vec.length)),
		minSizeInstance: Math.min(...wordVectorsOriginal.map(vec => vec.length)),
		wordCounts: tokenizer.wordCounts,
		/*wordIndex: tokenizer.wordIndex,*/
		/*indexWord: tokenizer.indexWord*/
	};
}

async function checkDataTokens(documents)
{
	let wordVectorsOriginal = myTokenizer.textsToSequences(documents.map(doc => doc.text));
	for(const vec of wordVectorsOriginal)
	{
		console.log(vec);
		await askQuestion("Le prochain? ");
	}

}

function askQuestion(query)
{
	const rl = readline.createInterface({
		input: process.stdin,
		output: process.stdout,
	});

	return new Promise(resolve => rl.question(query, ans => {
		rl.close();
		resolve(ans);
	}))
}

function getInstancesPerLabel(labels)
{
	return labels.reduce((total, currentValue, currentIndex, arr) =>
	{

		if(currentValue[0] === 1)
		{
			total[0]++;
		}
		else if(currentValue[1] === 1)
		{
			total[1]++;
		}
		else if(currentValue[2] === 1)
		{
			total[2]++;
		}
		return total;
	}, [0, 0, 0]);
}

function padSequences(sequences, maxLen, padding = 'post', truncating = 'pre', value = 0)
{
	return sequences.map(seq =>
	{
		// Perform truncation.
		if (seq.length > maxLen) {
			if (truncating === 'pre')
			{
				seq.splice(0, seq.length - maxLen);
			}
			else
			{
				seq.splice(maxLen, seq.length - maxLen);
			}
		}

		// Perform padding.
		if (seq.length < maxLen)
		{
			const pad = [];
			for (let i = 0; i < maxLen - seq.length; ++i)
			{
				pad.push(value);
			}
			if (padding === 'pre')
			{
				seq = pad.concat(seq);
			}
			else
			{
				seq = seq.concat(pad);
			}
		}

		return seq;
	});
}