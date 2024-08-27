import fiftyone as fo

datasetName = "06_V7"
dataset = fo.load_dataset(datasetName)
tagsInDataset = ["Van need label"]
viewMin = 0
viewMax = 10

anno_key = "VanV6_GTV1"
classesForAnnotations = ["car"]

def CreateView(dataset, tagsInDataset, viewMin, viewMax):
    if (viewMax - viewMin > 100):
        print("View range more that label studio can take")
        return
    view = (
        dataset
        .match_tags(tagsInDataset, bool=True)
    )
    view = view[viewMin:viewMax]
    return view

view = CreateView(dataset, tagsInDataset, viewMin, viewMax)

def LaunchLabelStudio(anno_key, view, classesForAnnotations):
    view.annotate(
        anno_key,
        backend="labelstudio",
        label_field="GTV1_BL1_101COCO",
        label_type="detections",
        classes = classesForAnnotations,
        launch_editor=True,
        url="http://localhost:8081",  
    )

LaunchLabelStudio(anno_key, view, classesForAnnotations)