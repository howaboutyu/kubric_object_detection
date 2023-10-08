# Synthetic data generation 

This project is to generate synthetic data for object detection. The goal is to generate a dataset that can be used to train an object detector to detect bottles or cans.

## Tools 

Several synthetic data generation tools are available. After looking at a few - including nvidia omniverse, unity - Kubric seems like a good choice. 

Nvidia Omniverse requires a RTX card and I don't have one locally. Kubric is a python library developed by google and is easy to setup using docker.


[link to Kubric](https://github.com/google-research/kubric/blob/main/challenges/multiview_matting/worker.py)


## Kubric output

Classes: 

bottles - 1
cans - 2

## Tasks
### Kubric 

* [ ] create initial script with random objects on a plane
* [ ] add random lighting, camera fov
* [ ] add random motion blur
* [ ] add random lightin
* [ ] add random distractors
* [ ] random background using hdri domes
* [ ] add textures/objects of cans and bottles

## Kubric to tfrecords

* [ ] create tfrecords from kubric data


