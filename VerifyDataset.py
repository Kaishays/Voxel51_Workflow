import fiftyone as fo
from fiftyone import ViewField as F

dataset = fo.load_dataset("06_V3")
label_field = "ground_truths"
view = (
    dataset
    .match_tags(["train"], bool=True, all = True)
    .match(F(label_field).exists())
)


count = 0
for sample in view:
    count += 1

print(count)

#verify each image is 640x640 max
#verify each image has label 



