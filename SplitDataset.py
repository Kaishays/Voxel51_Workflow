import fiftyone as fo
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


dataset = fo.load_dataset("06_V7")
 
IouDetField = "GTV1_BL1_101COCO"  
className = "car"  
saved_view_name = "GTV1_BL1_101COCO Final"
view = dataset.load_saved_view(saved_view_name)


samples_tensor = torch.tensor(np.array(list(view)), device='cuda')

dataset = TensorDataset(samples_tensor)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

shuffled_samples = []

count = 0
for batch in tqdm(dataloader, total=len(dataloader)):
    count += 1
    print(count)
    shuffled_samples.append(batch[0].cpu())  # Move batch to CPU and append

# Concatenate all batches into one tensor
shuffled_samples = torch.cat(shuffled_samples)

shuffled_samples_list = shuffled_samples.numpy().tolist()



num_samples = len(shuffled_samples_list)
num_train = int(0.7 * num_samples)
num_val = int(0.2 * num_samples)
num_final = num_samples - num_train - num_val

count = 0
for idx, sample in enumerate(shuffled_samples_list):
    if idx < num_train:
        sample.tags.append("train")
    elif idx < num_train + num_val:
        sample.tags.append("val")
    else:
        sample.tags.append("final")
    count += 1
    print(count)
    sample.save()

print("Dataset samples have been tagged.")
#dataset.untag_samples("name of tag")