import fiftyone as fo

datasetName = "06_V3"
dataset = fo.load_dataset(datasetName)
anno_key = "Car_101_200"

def LoadAnnotationsIntoDataset():
    dataset.load_annotations(
        anno_key,
        url="http://localhost:8080", 
    )
    return dataset

def LaunchLabelStudio():
    if __name__ == "__main__":
        session = fo.launch_app(dataset)
        session.wait()

def CleanLabelingJob():
    #results = dataset.load_annotation_results(anno_key)
    #results.cleanup()
    #dataset.delete_annotation_run(anno_key)
    print("place keeper")

dataset = LoadAnnotationsIntoDataset()
LaunchLabelStudio()
CleanLabelingJob()