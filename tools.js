import filesSystem from "fs";

export function writeJSONFile(data, path)
{
	filesSystem.writeFileSync(path, JSON.stringify(data, null, 4), "utf8");
}

export function deleteExtension(x)
{
	return x.replace(/\.[^/.]+$/, "");
}