# telexistence_assignment

Tasks: 
Use synthetic data to train an object detector used to detect bottles or cans.

This project can be broken down into two parts; synthetic data generation and object detection.



Several challenges are presented: 

* Different cameras, lighting, backgrounds, specular highlights
* Motion blur
* Has objects that are not bottles or cans

If we are using synthetic data it won't have any idea of 

### Ideas for synthetic data generation

* Kubric - mask/bb data generation
* Generative AI 
    ** texture re/generation 
    ** mesh generation


### Object detection

* effdet tf object detection API


## Plan 

* generate quick synthetic dataset from the `assets` folder using kubric and train an model with effdet to see


## Tasks

[ ] synthetic data generation
[ ] object detection
[ ] tests 
[ ] report