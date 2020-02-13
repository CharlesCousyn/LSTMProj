import * as tensorflow from '@tensorflow/tfjs-node-gpu'
import {jsonDocumentsToTensors} from "./dataUtils";
import GENERAL_CONFIG from "./configFiles/generalConfig";
import * as DATASET_INIT from "./datasetInit";
import csvToJson from "csv-file-to-json";
import ConfusionMatrix from 'ml-confusion-matrix';


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

    let dataTensorsTest2 = dataTensorsTest/*.slice([0], [4])*/;
    let labelTensorsTest2 = labelTensorsTest/*.slice([0], [4])*/;

    const MODEL = await tensorflow.loadLayersModel(`file://goodModel/model.json`);
    console.log(dataTensorsTest2.shape);
    const predictions = MODEL.predict(dataTensorsTest2);


    const trueLabels = labelNNToLabelString(oneDTo2D(Array.from(labelTensorsTest2.dataSync())));
    console.log("trueLabels", trueLabels);
    const predictedLabels = labelNNToLabelString(oneDTo2D(Array.from(predictions.dataSync())).map(floatToIntTAb));
    console.log("predictedLabels", predictedLabels);
    //const trueLabels =      [0, 1, 0, 1, 1, 0, 1, 1];
    //const predictedLabels = [1, 1, 1, 1, 0, 0, 1, 1];

    const CM2 = ConfusionMatrix.fromLabels(trueLabels, predictedLabels);
    console.log(CM2);
    console.table(CM2.matrix);
    console.log(CM2.getAccuracy());
})();

function oneDTo2D(array)
{
    let newArr = [];
    while(array.length)
    {
        newArr.push(array.splice(0,3));
    }
    return newArr;
}

function labelNNToLabelString(array)
{
    return array.map(el =>
    {
        if(el[0] === 1)
        {
            return "J";
        }
        if(el[1] === 1)
        {
            return "E";
        }
        if(el[2] === 1)
        {
            return "C";
        }
    });
}

function floatToIntTAb(array)
{
    let res = [0, 0, 0];
    let i = array.indexOf(Math.max(...array));
    res[i] = 1;
    return res;
}