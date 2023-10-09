# Telexistence Assignment Readme

## Introduction

This repository contains the code and resources for an object detection project focused on identifying bottles or cans using synthetic data. The project can be divided into two main components: synthetic data generation and object detection.

## Challenges

Several challenges are encountered in this project, including:

* Variation in cameras, lighting conditions, backgrounds, and specular highlights.
* Motion blur in the images.
* Presence of objects other than bottles or cans.
* Different environmental settings, such as items inside a fridge, on a table, or on a shelf.
* Diverse material textures, including glass, plastic, and metal.
* Varied object shapes, such as round, square, and cylindrical.

## Synthetic Data Generation Ideas

To address these challenges, consider the following ideas for synthetic data generation:

* Utilize Kubric for mask and bounding box data generation.
* Explore Generative AI techniques for texture generation and regeneration.
* Create synthetic object meshes to diversify the dataset.

## Object Detection Approach

For object detection, the project employs the `effdet` TensorFlow object detection API.

## Project Plan

The project plan is as follows:

1. Generate a preliminary synthetic dataset using the assets provided in the `assets` folder.
2. Utilize Kubric for mask and bounding box data generation.
3. Train a model using the TensorFlow object detection API.
4. Evaluate the model's performance with appropriate tests.
5. Document the findings and outcomes in a report.
