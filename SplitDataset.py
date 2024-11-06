import fiftyone as fo
import fiftyone.utils.random as four


dataset = fo.load_dataset("06_V7")
 




IouDetField = "GTV1_BL1_101COCO"  
className = "car"  
saved_view_name = "GTV1_BL1_101COCO Final"
view = dataset.load_saved_view(saved_view_name)

#view.untag_labels("val", "sample tags")
#view.untag_labels("test", "sample tags")

num_samples = len(view)

four.random_split(view, {"trainV2": 0.7, "testV2": 0.2, "valV2": 0.1})

''''
num_samples = len(view)
num_train = int(0.7 * num_samples)
num_val = int(0.2 * num_samples)
num_final = num_samples - num_train - num_val

count = 0
for idx, sample in enumerate(view):
    if idx < num_train:
        sample.tags.append("trainv2")
    elif idx < num_train + num_val:
        sample.tags.append("valv2")
    else:
        sample.tags.append("finalv2")
    count += 1
    print(count)
    sample.save()

print("Dataset samples have been tagged.")
#dataset.untag_samples("name of tag")
'''