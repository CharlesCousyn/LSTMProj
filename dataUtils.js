import * as DATASET_INIT from "./datasetInit.js"
import GENERAL_CONFIG from "./configFiles/generalConfig"
import tokenizer from "./tokenizer.js"
import * as tensorflow from "@tensorflow/tfjs-node-gpu"
import csvToJson from "csv-file-to-json"

export let myTokenizer = null;

function initTokenizer(documents)
{
	myTokenizer = new tokenizer();
	const complaintsTexts = documents.map(doc => doc.text);
	myTokenizer.fitOnTexts(complaintsTexts);
}

export function jsonDocumentsToTensors(documents, sequenceSizePerDocument)
{
	initTokenizer(documents);

	let wordVectorsOriginal = myTokenizer.textsToSequences(documents.map(doc => doc.text));
	const labels = documents.map(doc => doc.labelNN);

	//Keeping maxWordsPerDocument elements per document and adding 0 when too short
	wordVectorsOriginal = padSequences(wordVectorsOriginal, sequenceSizePerDocument, "post", "pre", 0);

	/*
	//Split train and test dataset
	const indexOfSplit = Math.round(validationSplit * wordVectorsOriginal.length);
	const wordVectorsOriginalTrain = wordVectorsOriginal.slice(0, indexOfSplit);
	const wordVectorsOriginalTest = wordVectorsOriginal.slice(indexOfSplit, wordVectorsOriginal.length);
	const labelsTrain = labels.slice(0, indexOfSplit);
	const labelsTest = labels.slice(indexOfSplit, labels.length);*/

	return {
		dataTensors: tensorflow.tensor(wordVectorsOriginal),
		labelTensors: tensorflow.tensor(labels),
	};
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