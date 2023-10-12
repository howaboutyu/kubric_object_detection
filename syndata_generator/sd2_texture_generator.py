import os
import torch
import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

def generate_image(pipe, prompt, output_filename):
    image = pipe(prompt, num_inference_steps=25).images[0]
    image.save(output_filename)

def main(args):
    # Initialize the DiffusionPipeline
    repo_id = "stabilityai/stable-diffusion-2-base"
    pipe = DiffusionPipeline.from_pretrained(
        repo_id, torch_dtype=torch.float16, revision="fp16"
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    # Directory to save the generated images
    output_dir = "generated_textures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate and save the images
    for i in range(args.num_images):
        output_filename = os.path.join(output_dir, f"texture_{i+1}.png")
        print(f"Generating texture for: {args.prompt}")
        generate_image(pipe, args.prompt, output_filename)

    print(f"{args.num_images} textures generated and saved in '{output_dir}' folder.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate texture images from a given prompt.')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to use for generating the texture image.')
    parser.add_argument('--output_folder', type=str, required=True, help='The folder to save the texture image.')
    parser.add_argument('--num_images', type=int, default=10000, help='Number of images to generate.')

    args = parser.parse_args()

    main(args)
