import fiftyone as fo

dataset = fo.load_dataset("06_V4")

label_field = "ground_truths"  

samples_to_delete = []
count = 0
for sample in dataset:
    if label_field in sample and sample[label_field] is not None:
        count += 1
        print(count)
        samples_to_delete.append(sample.id)

if samples_to_delete:
    print(f"Deleted {len(samples_to_delete)} samples from the dataset '{dataset.name}' that contained the field '{label_field}'.")
    dataset.delete_samples(samples_to_delete)
else:
    print(f"No samples found with the field '{label_field}' in the dataset '{dataset.name}'.")

dataset.save()