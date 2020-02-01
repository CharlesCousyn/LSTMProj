import filesSystem from "fs";
import * as TOOLS from "./tools.js"

export function filesDocumentToJsonDocument(pathToDataFileDocuments, pathToFolderOfTextFiles, jsonObjectWithLabels, labelMapping)
{
	//Check if file already exist
	if(!filesSystem.existsSync(pathToDataFileDocuments))
	{
		const documentObjects = filesSystem.readdirSync(pathToFolderOfTextFiles, {encoding:"utf8", withFileTypes: true})
		.filter(dirent => dirent.isFile())
		.map(dirent => ({name: TOOLS.deleteExtension(dirent.name).toLowerCase(), path: `${pathToFolderOfTextFiles}/${dirent.name}`, text: null, label: null}))
		.map(addTextFromOneFile)
		.map(documentObject => addLabels(documentObject, jsonObjectWithLabels))
		.map(documentObject => addLabelsForNN(documentObject, labelMapping));

		TOOLS.writeJSONFile(documentObjects, pathToDataFileDocuments);
	}

	return JSON.parse(filesSystem.readFileSync(pathToDataFileDocuments, {encoding: "utf8"}));
}

function addTextFromOneFile(documentObject)
{
	documentObject.text = filesSystem.readFileSync(documentObject.path, {encoding: "utf8"});
	return documentObject;
}

function addLabels(documentObject, objectWithLabels)
{
	try
	{
		documentObject.label = objectWithLabels.find(complaintObj => complaintObj.Complaints.toLowerCase() === documentObject.name).Types;
	}
	catch(e)
	{
		console.log(`Impossible to find document ${documentObject.name}`);
	}

	return documentObject;
}

function addLabelsForNN(documentObject, labelMapping)
{
	let labTab = new Array(Object.keys(labelMapping).length).fill(0);
	labTab[labelMapping[documentObject.label]] = 1;
	documentObject.labelNN = labTab;

	return documentObject;
}

