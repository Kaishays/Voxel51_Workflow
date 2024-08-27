import fiftyone as fo
import os

dirToSrchForJpg = "C:/Git/ml/DataManagement/datasets/06All/06_JPGs"
export_dir = "C:/Git/ml/DataManagement/datasets/voxelExport3_756,000"
datasetName = "06_V4"

def FindAllJpgs(directoryToSearchForJpg):
    filepaths = []
    for root, dirs, files in os.walk(directoryToSearchForJpg):
        for file in files:
            if file.endswith(".jpg"):
                filepaths.append(os.path.join(root, file))
    return filepaths

def CreateDataset(datasetName):
    dataset = fo.Dataset(datasetName)
    dataset = fo.load_dataset(datasetName)
    dataset.persistent = True
    dataset.media_type = "image"
    
def AddSamplesToDataset(dataset, filepaths):   
    for filepath in filepaths:
        sample = fo.Sample(filepath=filepath, media_type="image")
        #sample["image"] = fo.Classification(label="Flight_Frame")
        dataset.add_sample(sample)
        print(dataset)
    return dataset

def LaunchDataset(dataset):
    if __name__ == "__main__":
        session = fo.launch_app(dataset)
        session.wait()

filePaths = FindAllJpgs(dirToSrchForJpg)
dataset = CreateDataset(datasetName)
AddSamplesToDataset(dataset, filePaths)
LaunchDataset(dataset)