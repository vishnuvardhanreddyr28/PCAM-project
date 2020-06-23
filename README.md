# PCAM-project
The PatchCamelyon benchmark is a new and challenging image classification dataset. It consists of 327.680 color images (96 x 96px) extracted from histopathologic scans of lymph node sections. Each image is annoted with a binary label indicating presence of metastatic tissue. PCam provides a new benchmark for machine learning models: bigger than CIFAR10, smaller than imagenet, trainable on a single GPU.

The above description and data is avilable in the following link https://github.com/basveeling/pcam

I have used keras and tensorflow as backend.Since data is very large you might need a HPC to train the model.Alternatively you can also use divide the images in batches and train your model.
