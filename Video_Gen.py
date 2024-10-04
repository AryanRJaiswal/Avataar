import argparse
from PIL import Image, ImageEnhance, ImageFilter
import torch
from diffusers import StableDiffusionPipeline
import rembg
import io
import random
import numpy as np
import cv2
from tqdm import tqdm

def adjust_object(obj_img, background):
    # Randomly adjust brightness and contrast for better blending
    brightness_adj = random.uniform(0.9, 1.1)
    contrast_adj = random.uniform(0.9, 1.1)
    
    # Apply brightness and contrast changes
    obj_img = ImageEnhance.Brightness(obj_img).enhance(brightness_adj)
    obj_img = ImageEnhance.Contrast(obj_img).enhance(contrast_adj)
    
    # Apply slight blur for a more natural look
    obj_img = obj_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return obj_img

def apply_shadow(background, obj_img, position):
    # Create a semi-transparent shadow with a slight blur
    shadow_layer = Image.new("RGBA", obj_img.size, (0, 0, 0, 100))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=5))
    
    # Offset the shadow position slightly
    shadow_position = (position[0] + 5, position[1] + 5)
    
    # Add the shadow to the background
    background.paste(shadow_layer, shadow_position, shadow_layer)
    
    return background

def merge_images(background, obj_img, scale, position):
    # Resize the object based on the scaling factor
    resized_obj = obj_img.copy()
    resized_obj.thumbnail((int(resized_obj.width * scale), int(resized_obj.height * scale)), Image.Resampling.LANCZOS)
    
    # Adjust the object's appearance for better background integration
    resized_obj = adjust_object(resized_obj, background)
    
    # Apply shadow to the object
    background = apply_shadow(background, resized_obj, position)
    
    # Place the object onto the background
    background.paste(resized_obj, position, resized_obj)
    
    return background

def create_frame(pipe, obj_img, prompt, width, height, zoom):
    # Ensure dimensions are multiples of 8 for compatibility
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    # Generate background using Stable Diffusion with the provided prompt
    bg_image = pipe(prompt, height=height, width=width).images[0].convert("RGBA")
    
    # Calculate scaling and positioning for the object
    scale = (width / 4) / obj_img.width * zoom
    position = (int(width * 0.7 * zoom), int(height * 0.6 * zoom))
    
    # Merge the object into the generated background
    final_frame = merge_images(bg_image, obj_img, scale, position)
    
    return final_frame

def save_video(frames, filepath, fps=24):
    # Determine the resolution from the first frame
    frame_height, frame_width = np.array(frames[0]).shape[:2]
    
    # Set up video writer with MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))

    # Write frames to the video
    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    
    # Release the video file
    video.release()

def create_zoom_video(image_path, prompt, save_path, total_frames=60):
    # Open the image and remove its background
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        obj_img_data = rembg.remove(img_data)
        obj_img = Image.open(io.BytesIO(obj_img_data)).convert("RGBA")

    # Load the Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize frame dimensions
    base_width, base_height = 768, 512
    frames = []

    # Generate video frames with a zoom-out effect
    for i in tqdm(range(total_frames), desc="Creating frames"):
        zoom_factor = 1 + (i / total_frames)
        current_width = int(base_width * zoom_factor)
        current_height = int(base_height * zoom_factor)
        
        frame = create_frame(pipeline, obj_img, prompt, current_width, current_height, zoom_factor)
        frame = frame.resize((base_width, base_height), Image.Resampling.LANCZOS)
        frames.append(frame)

    # Compile frames into a video
    save_video(frames, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a zooming video with an object blended into a generated background.")
    parser.add_argument("--image", required=True, help="Path to the object image.")
    parser.add_argument("--text-prompt", required=True, help="Text prompt for generating the background.")
    parser.add_argument("--output", required=True, help="Output path for the generated video.")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames for the video (default: 60)")

    args = parser.parse_args()

    create_zoom_video(args.image, args.text_prompt, args.output, args.frames)
