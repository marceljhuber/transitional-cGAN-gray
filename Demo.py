#!/usr/bin/env python
# coding: utf-8

# # Generating 4 classes

# In[1]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import torch
import legacy
import dnnlib
import numpy as np
import tqdm.notebook as tqdm
import imageio
import shutil
import os
from PIL import Image
import gradio as gr
import subprocess
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Path to the trained model
model_path = os.path.join(os.getcwd(), "models", "oct.pkl") # ["NORMAL", "CNV", "DRUSEN", "DME"]
model_path = os.path.join(os.getcwd(), "models", "oct_256_resized_1600.pkl") # ["DRUSEN", "NORMAL", "DME", "CNV"]

# Directory where temporary files should be stored
directory = os.path.join(os.getcwd(), "demo")

# Check if the directory exists
if os.path.exists(directory):
    # If it exists, delete it and its contents
    shutil.rmtree(directory)

# Create the directory
os.makedirs(directory)

# Available classes to choose from
classes = ["DRUSEN", "NORMAL", "DME", "CNV"]
classes_dict = {"DRUSEN": 0, "NORMAL": 1, "DME": 2, "CNV": 3}

# Torch device
device = torch.device('cuda')


# # Generate an image

# In[3]:


# Method to generate an image by choosing out of four classes 
def generate_noise_image(choice):
    random_seed = np.random.randint(1, 9999)
    execution = f"python generate.py --outdir={directory} --seeds={random_seed} --network={model_path} --class={classes_dict[choice]} --vector-mode=True"
    os.system(execution)    
    img_path = os.path.join(os.getcwd(), directory, f"{classes_dict[choice]}_{str(random_seed).zfill(4)}.png")
    img = Image.open(img_path)
    return img

iface1 = gr.Interface(
    fn=generate_noise_image,
    inputs=gr.inputs.Dropdown(choices=classes, default="NORMAL", label="Class"),
    outputs=gr.outputs.Image(type="pil").style(height=256, width=256)
)


# # Interpolating between x images (z-vectors)

# In[4]:


def combine_images(image1_path, image2_path, output_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Determine the maximum width and total height
    max_width = max(image1.width, image2.width)
    total_height = max(image1.height, image2.height)

    # Create a new blank image with the combined size
    combined_image = Image.new("RGB", (max_width * 2, total_height))

    # Paste the first image on the left side
    combined_image.paste(image1, (0, 0))

    # Paste the second image on the right side
    combined_image.paste(image2, (max_width, 0))

    # Save the combined image
    combined_image.save(output_path)


# In[5]:


# Method to create a video of an interpolation between x images
def generate_noise_image(sequence, frames):
    try:
        shutil.rmtree(directory)
    except OSError as error:
        print(f"Failed to delete directory '{directory}': {error}")
    
    counts = [sequence.count(str(i)) for i in range(4)]
    random_seeds = np.sort(np.random.randint(1, 9999, sum(counts)))
    lvecs = []
    generated_images = []  # Store the generated images
    
    for i in range(sum(counts)):
        execution = f"python generate.py --outdir={directory} --seeds={random_seeds[i]} --network={model_path} --class={sequence[i]} --vector-mode=True"
        os.system(execution)
        
        generated_image_path = os.path.join(directory, f"{sequence[i]}_{str(random_seeds[i]).zfill(4)}.png")
        
        # Read the image using PIL
        # image = Image.open(generated_image_path)
        
        # Convert the image to a NumPy array
        # image_array = np.array(image)
        
        generated_images.append(generated_image_path)
        
        ################################################################################################################
        class_idx = int(sequence[i])

        with dnnlib.util.open_url(model_path) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

        @torch.no_grad()
        def map_z_to_w(z, G):
            label = torch.zeros((1, G.c_dim), device=device)
            label[:, class_idx] = 1
            w = G.mapping(z.to(device), label)
            return w

        # Load z from file.
        z_path = os.path.join(os.getcwd(), "demo", f"{sequence[i]}_{str(random_seeds[i]).zfill(4)}.npy")
        z_np = np.load(z_path)

        # Convert `z_np` to a PyTorch tensor.
        z = torch.as_tensor(z_np, device=device)

        # Convert z to w.
        w = map_z_to_w(z, G)
        ################################################################################################################
        
        lvecs.append(w)

    FPS = 60
    FREEZE_STEPS = 30
    STEPS = int(frames)
    
    with dnnlib.util.open_url(model_path) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore
    
    video = imageio.get_writer(f'{directory}/video.mp4', mode='I', fps=FPS, codec='libx264', bitrate='16M')

    for i in range(sum(counts) - 1):# load z_arr from npz file
        diff = lvecs[i+1] - lvecs[i]
        step = diff / STEPS
        current = lvecs[i].clone()
        target_uint8 = np.array([256, 256, 3], dtype=np.uint8)


        for j in range(STEPS):
            z = current.to(device)
            synth_image = G.synthesis(z, noise_mode='const')
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            repeat = FREEZE_STEPS if j == 0 or j == (STEPS - 1) else 1

            for i in range(repeat):
                video.append_data(synth_image)
            current = current + step

    video.close()
    
    out_img = os.path.join(directory, 'combined.png')
    combine_images(generated_images[0], generated_images[1], out_img)
    
    vid = os.path.join(directory, 'video.mp4')
    img = os.path.join(directory, 'combined.png')
    
    return vid, img #generated_images[0], generated_images[1]

sequence = gr.inputs.Textbox(default="13", label="Sequence input")
frames = gr.inputs.Number(default=240, label="Frames")
output_video = gr.outputs.Video(type="mp4").style(height=256, width=256)
output_image = gr.outputs.Image(type="filepath", label="Generated Images").style(height=256, width=512)
# output_image1 = gr.outputs.Image(type="numpy", label="Generated Image 1")
# output_image1 = gr.outputs.Image(type="filepath", label="Generated Image 1").style(height=256, width=256)
# output_image2 = gr.outputs.Image(type="filepath", label="Generated Image 2").style(height=256, width=256)

# output = gr.outputs.Video(type="numpy").style(height=256, width=256)

iface2 = gr.Interface(
    fn=generate_noise_image,
    inputs=[sequence, frames],
    outputs=[output_video, output_image],
    description="This Gradio interface generates a video by morphing in the latent space between given class images. To use it, enter a string consisting only of letters from 0-3 to serve as class labels."
)


# # Interpolating between x images (approximation)

# In[6]:


# Method to create a video of an interpolation between x images
def generate_noise_image(sequence, frames, n_steps):
    try:
        shutil.rmtree(directory)
    except OSError as error:
        print(f"Failed to delete directory '{directory}': {error}")
    
    counts = [sequence.count(str(i)) for i in range(4)]
    random_seeds = np.sort(np.random.randint(1, 9999, sum(counts)))
    lvecs = []
    
    for i in range(sum(counts)):
        execution = f"python generate.py --outdir={directory} --seeds={random_seeds[i]} --network={model_path} --class={sequence[i]}"
        os.system(execution)
        
        outdir_path = os.path.join(directory, "projections", str(i).zfill(2))
        target_dir = os.path.join(directory, f"{sequence[i]}_{str(random_seeds[i]).zfill(4)}.png")
        execution = f"python projector.py --outdir={outdir_path} --target={target_dir} --network={model_path} --num-steps={int(n_steps)} --save-video=True"
        os.system(execution)
        
        lvec_path = os.path.join(os.getcwd(), directory, "projections", f"{str(i).zfill(2)}", "projected_w.npz") 
        lvecs.append(np.load(lvec_path)['w'])

    FPS = 60
    FREEZE_STEPS = 30
    STEPS = int(frames)
    
    with dnnlib.util.open_url(model_path) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore
    
    video_path = os.path.join(os.getcwd(), directory, "video.mp4")
    video = imageio.get_writer(video_path, mode='I', fps=FPS, codec='libx264', bitrate='16M')

    for i in range(sum(counts) - 1):
        diff = lvecs[i+1] - lvecs[i]
        step = diff / STEPS
        current = lvecs[i].copy()
        target_uint8 = np.array([256, 256, 3], dtype=np.uint8)


        for j in range(STEPS):
            z = torch.from_numpy(current).to(device)
            synth_image = G.synthesis(z, noise_mode='const')
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            repeat = FREEZE_STEPS if j == 0 or j == (STEPS - 1) else 1

            for i in range(repeat):
                video.append_data(synth_image)
            current = current + step

    video.close()
    
    return os.path.join(os.getcwd(), directory, "video.mp4")

sequence = gr.inputs.Textbox(default="13", label="Sequence input")
n_steps = gr.inputs.Number(default=100, label="Latent Steps")
frames = gr.inputs.Number(default=240, label="Frames")
output = gr.outputs.Video(type="mp4").style(height=256, width=256)

iface3 = gr.Interface(
    fn=generate_noise_image,
    inputs=[sequence, frames, n_steps],
    outputs=output,
    description="This Gradio interface generates a video by morphing in the latent space between given class images. To use it, enter a string consisting only of letters from 0-3 to serve as class labels."
)


# # Generate variations of an image

# In[7]:


# Method to create a video of an interpolation between x images
def generate_noise_image(sequence, frames):
    try:
        shutil.rmtree(directory)
    except OSError as error:
        print(f"Failed to delete directory '{directory}': {error}")
    
    counts = [sequence.count(str(i)) for i in range(4)]
    random_seeds = [np.random.randint(1, 9999)] * sum(counts)
    lvecs = []
    
    for i in range(sum(counts)):
        execution = f"python generate.py --outdir={directory} --seeds={random_seeds[i]} --network={model_path} --class={sequence[i]} --vector-mode=True"
        # os.system(execution)
        subprocess.run(execution, shell=True)
        
        ################################################################################################################
        class_idx = int(sequence[i])

        with dnnlib.util.open_url(model_path) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

        @torch.no_grad()
        def map_z_to_w(z, G):
            label = torch.zeros((1, G.c_dim), device=device)
            label[:, class_idx] = 1
            w = G.mapping(z.to(device), label)
            return w

        # Load z from file.
        # z_path = os.path.join(os.getcwd(), "demo", f"seed{str(random_seeds[i]).zfill(4)}.npy")
        z_path = os.path.join(os.getcwd(), "demo", f"{sequence[i]}_{str(random_seeds[i]).zfill(4)}.npy")
        z_np = np.load(z_path)

        # Convert `z_np` to a PyTorch tensor.
        z = torch.as_tensor(z_np, device=device)

        # Convert z to w.
        w = map_z_to_w(z, G)
        ################################################################################################################
        
        lvecs.append(w)

    FPS = 60
    FREEZE_STEPS = 30
    STEPS = int(frames)
    
    with dnnlib.util.open_url(model_path) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore
    
    video = imageio.get_writer(f'{directory}/video.mp4', mode='I', fps=FPS, codec='libx264', bitrate='16M')

    for i in range(sum(counts) - 1):# load z_arr from npz file
        diff = lvecs[i+1] - lvecs[i]
        step = diff / STEPS
        current = lvecs[i].clone()
        target_uint8 = np.array([256, 256, 3], dtype=np.uint8)


        for j in range(STEPS):
            z = current.to(device)
            synth_image = G.synthesis(z, noise_mode='const')
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            repeat = FREEZE_STEPS if j == 0 or j == (STEPS - 1) else 1

            for i in range(repeat):
                video.append_data(synth_image)
            current = current + step

    video.close()
    
    return f"{directory}/video.mp4"

sequence = gr.inputs.Textbox(default="13", label="Sequence input")
frames = gr.inputs.Number(default=240, label="Frames")
output = gr.outputs.Video(type="mp4").style(height=256, width=256)

iface4 = gr.Interface(
    fn=generate_noise_image,
    inputs=[sequence, frames],
    outputs=output,
    description="This Gradio interface generates a video by morphing in the latent space between given class images. To use it, enter a string consisting only of letters from 0-3 to serve as class labels."
)


# # Approximate a given image in the latent space

# In[8]:


def edit_image(input_image, num_steps=1000):
    pil_image = Image.fromarray(input_image)
    pil_image.save(os.path.join(directory, 'input_image.png'))
    execution = f"python projector.py --outdir={directory} --network={model_path} --target={os.path.join(directory, 'input_image.png')} --num-steps={int(num_steps)}"
    os.system(execution)
    
    # return execution #os.path.join(directory, 'proj.png')
    return os.path.join(directory, 'proj.png'), os.path.join(directory, 'proj.mp4')

steps = gr.inputs.Number(default=1000, label="Steps")
input_image = gr.outputs.Image(type="numpy").style(height=256, width=256)
output_image = gr.outputs.Image(type="pil").style(height=256, width=256)
output_video = gr.outputs.Video(type="mp4").style(height=256, width=512)

iface5 = gr.Interface(
    fn=edit_image, 
    inputs=[input_image, steps], 
    input_names=["Input Image", "Number of Steps"],
    input_labels=["Choose an image", "Number of Steps"],
    
    outputs=[output_image, output_video],
    output_names=["Edited Image", "Video File"],
    output_labels=["Edited Image", "Video File"],
    input_is_default=[True, False],
    description="This Gradio interface generates a latent approximation using an input image.")


# # Multiple Tabs

# In[9]:


gr.TabbedInterface(
    [iface1, iface2, iface3, iface4, iface5], ["Generate an image", "Generate a video (original)", "Generate a video (approximated)", "Generate variations of an image", "Project an image"]
).launch(share=True)

