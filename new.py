import fiftyone.zoo as foz

# Download only (without loading into FiftyOne)
foz.download_zoo_dataset("coco-2014", split="train")

# Or download and load into FiftyOne directly
import fiftyone as fo

dataset = foz.load_zoo_dataset("coco-2014", split="train")
session = fo.launch_app(dataset)