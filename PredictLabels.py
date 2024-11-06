import fiftyone as fo
import fiftyone.utils.annotations as foua
import onnxruntime as ort
import numpy as np
import os
from PIL import Image
from torchvision.transforms import functional as func
import torch
import torchvision





def FindAllJpgs(directoryToSearchForJpg):
    filepaths = []
    for root, dirs, files in os.walk(directoryToSearchForJpg):
        for file in files:
            if file.endswith(".jpg"):
                filepaths.append(os.path.join(root, file))
    return filepaths

def CreateDataset(datasetName):
    IMAGES_DIR = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/Train_Val_Final_AllTogether640/runs/slice_coco/labels_images_640_001"
    LABELS_PATH = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/Train_Val_Final_AllTogether640/runs/slice_coco/labels_640_001.json"

# Create a FiftyOne dataset from the COCO-style dataset directory
    dataset = fo.Dataset.from_dir(
        name=datasetName,  # Specify the dataset name
        dataset_type=fo.types.COCODetectionDataset,  # Specify the dataset type
        data_path=IMAGES_DIR,  # Path to the image directory
        labels_path=LABELS_PATH,  # Path to the COCO annotations JSON file
        include_id=True,  # Include image IDs from the COCO dataset
    ) 
    dataset = fo.load_dataset(datasetName)
    dataset.persistent = True
    dataset.media_type = "image"

def AddSamplesToDataset(dataset, filepaths):   
    count = 0
    for filepath in filepaths:
        count+=1
        print(count)
        sample = fo.Sample(filepath=filepath, media_type="image")
        dataset.add_sample(sample)
    return dataset

def LaunchDataset(dataset):
    if __name__ == "__main__":
        session = fo.launch_app(dataset)
        session.wait()


























dirToSrchForJpg = "C:/Git/ml/DataManagement/datasets/06_V5_Temp/oneIN3"
datasetName = "06_V7"

#filePaths = FindAllJpgs(dirToSrchForJpg)
#print("Files found: " + str(len(filePaths)))
dataset = fo.load_dataset(datasetName)
#print(str(dataset.count))
#dataset = AddSamplesToDataset(dataset, filePaths)
#LaunchDataset(dataset)




model_path = "C:/Git/ml/models/OnnxModels/COCO/rtdetr_r101vd_2x_coco_objects365_from_paddle.onnx"
session = ort.InferenceSession(
        model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

COCO_Labels = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "TV",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]

def handle_labels_tensor(labels_tensor):
    """
    Handles a tensor of labels, converts indices to string labels using a label map.

    Args:
        labels_tensor (np.ndarray): A NumPy array representing the labels tensor.

    Returns:
        list: A list of string labels.
    """
    batch_size = labels_tensor.shape[0]  # Get the batch size
    number_of_predictions = labels_tensor.shape[1]  # Get the number of predictions per batch

    labels = []  # Initialize a list to store the resulting labels
    for i in range(batch_size):
        for j in range(number_of_predictions):
            label_index = labels_tensor[i, j]  # Get the label index
            labels.append(COCO_Labels[label_index])  # Map the index to a label string

    return labels

def handle_boxes_tensor(boxes_tensor):
    """
    Handles a tensor of bounding boxes and converts them to a list of rectangle representations.

    Args:
        boxes_tensor (np.ndarray): A NumPy array representing the bounding boxes tensor.

    Returns:
        list: A list of rectangle representations.
    """
    batch_size = boxes_tensor.shape[0]
    number_of_predictions = boxes_tensor.shape[1]

    bounding_boxes = []  # Initialize a list to store the resulting rectangles
    for batch_index in range(batch_size):
        for prediction_index in range(number_of_predictions):
            # Extract the coordinates for each bounding box
            top_left_x = float(boxes_tensor[batch_index, prediction_index, 0])
            top_left_y = float(boxes_tensor[batch_index, prediction_index, 1])
            bottom_right_x = float(boxes_tensor[batch_index, prediction_index, 2])
            bottom_right_y = float(boxes_tensor[batch_index, prediction_index, 3])
            
            # Convert coordinates to normalized format (x_min, y_min, width, height)
            width = bottom_right_x - top_left_x
            height = bottom_right_y - top_left_y
            x_min = top_left_x
            y_min = top_left_y
            
            # Normalize coordinates by image dimensions
            x_min /= 640  # Assuming image width is 640 pixels
            y_min /= 640  # Assuming image height is 640 pixels
            width /= 640
            height /= 640

            bounding_box = (x_min, y_min, width, height)  # Normalized bounding box

            bounding_boxes.append(bounding_box)

    return bounding_boxes
def handle_scores_tensor(scores_tensor):
    """
    Handles a tensor of scores, rounds each score to two decimal places, and returns a list of scores.

    Args:
        scores_tensor (np.ndarray): A NumPy array representing the scores tensor.

    Returns:
        list: A list of rounded float scores.
    """
    batch_size = scores_tensor.shape[0]  # Get the batch size
    number_of_scores = scores_tensor.shape[1]  # Get the number of scores per batch

    scores = []  # Initialize a list to store the resulting scores
    for i in range(batch_size):
        for j in range(number_of_scores):
            score = float(scores_tensor[i, j])  # Get the score
            rounded_value = round(score, 2)  # Round the score to two decimal places
            scores.append(rounded_value)  # Add the rounded score to the list

    return scores



def postprocess(outputs):
    boxes, labels, scores = outputs
    detections = []
    passedPostProcess = 0
    for label, box, score in zip(labels, boxes, scores):
        if score > 0.5:
            if label in ["car", "truck", "bus", "person"]:
                passedPostProcess += 1
                print("                                                Passed: " + str(passedPostProcess))
                detection = fo.Detection(label=label, bounding_box=box, confidence=score)
                detections.append(detection)
    return detections, passedPostProcess

def image_to_tensor_fp32(image):
    """
    Converts a PIL image to a tensor of shape (1, 3, height, width) with normalized float32 values.
    
    Args:
        image (PIL.Image.Image): The input image object.
        
    Returns:
        np.ndarray: A tensor of shape (1, 3, height, width).
    """
    image = image.convert('RGB')
    
    width, height = image.size
    
    pixels = np.array(image, dtype=np.float32) / 255.0
    
    red_channel = pixels[:, :, 0]
    green_channel = pixels[:, :, 1]
    blue_channel = pixels[:, :, 2]
    
    data = np.stack([red_channel, green_channel, blue_channel], axis=0)
    data = np.expand_dims(data, axis=0)  # Shape: (1, 3, height, width)
    
    return data
def create_origin_target_size_tensor(batch_size=1):
    """
    Creates a tensor of shape (batch_size, 2) with int64 type, initialized with the origin height and width.
    
    Args:
        origin_height (int): The height of the original image.
        origin_width (int): The width of the original image.
        batch_size (int): The batch size. Defaults to 1.
        
    Returns:
        np.ndarray: A NumPy array of shape (batch_size, 2) with the specified dimensions.
    """
    orig_target_sizes = np.zeros((batch_size, 2), dtype=np.int64)
    
    for i in range(batch_size):
        orig_target_sizes[i, 0] = 640
        orig_target_sizes[i, 1] = 640
    
    return orig_target_sizes

sampleInDatasetCount = 0
equalTo640 = 0
lessThan640 = 0
greaterThan640 = 0
passedPostProcess = 0
for sample in dataset:

    sampleInDatasetCount += 1
    print(sampleInDatasetCount)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_path = sample.filepath
    image = Image.open(image_path)
    imageWidth = image.width
    imageHeight = image.height
    if imageWidth == 640 and imageHeight == 640:
        equalTo640 += 1
        tensor = image_to_tensor_fp32(image)
        orginTargetSize = create_origin_target_size_tensor(batch_size=1)

        outputs = session.run(None, {"images": tensor, "orig_target_sizes": orginTargetSize})

        labels = handle_labels_tensor(outputs[0])  
        boxes = handle_boxes_tensor(outputs[1])
        scores = handle_scores_tensor(outputs[2])
        detections, passedPostProcess_ = postprocess([boxes, labels, scores])
        passedPostProcess = passedPostProcess_
        sample["RT-DETR_101_COCO"] = fo.Detections(detections=detections)
        sample.save()
    if imageWidth < 640 or imageHeight < 640:
        lessThan640 += 1
        print ("less than: " + str(lessThan640))
    if imageWidth > 640 > imageHeight < 640:
        greaterThan640 += 1

LaunchDataset(dataset)