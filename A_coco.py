import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("coco-2014", split="validation")
session = fo.launch_app(dataset)

session.wait()  # ← This keeps the app alive