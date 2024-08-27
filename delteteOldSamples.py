
import fiftyone as fo



dataset = fo.load_dataset("06_V4")


label_field = "ground_truths"  # Replace with the name of the label field to check



# Iterate through the samples and delete those with the specified field
samples_to_delete = []
count = 0
for sample in dataset:
    # Check if the sample has the specified label field
    if label_field in sample and sample[label_field] is not None:
        count += 1
        print(count)
        samples_to_delete.append(sample.id)

# Delete the samples
if samples_to_delete:
    print(f"Deleted {len(samples_to_delete)} samples from the dataset '{dataset.name}' that contained the field '{label_field}'.")
    dataset.delete_samples(samples_to_delete)
else:
    print(f"No samples found with the field '{label_field}' in the dataset '{dataset.name}'.")

# Save changes to the dataset
dataset.save()