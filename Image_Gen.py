import argparse
from PIL import Image, ImageEnhance, ImageFilter
import torch
from diffusers import StableDiffusionPipeline
import rembg
import io
import random

def adjust_image(obj_img, background):
    # Adjust brightness and contrast to blend with the background
    brightness_adj = random.uniform(0.9, 1.1)
    contrast_adj = random.uniform(0.9, 1.1)
    
    brightness_enhancer = ImageEnhance.Brightness(obj_img)
    obj_img = brightness_enhancer.enhance(brightness_adj)
    contrast_enhancer = ImageEnhance.Contrast(obj_img)
    obj_img = contrast_enhancer.enhance(contrast_adj)
    
    # Apply a slight blur to harmonize sharpness with the background
    obj_img = obj_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return obj_img

def add_shadow_effect(background, obj_img, position):
    # Create a semi-transparent shadow
    shadow_layer = Image.new("RGBA", obj_img.size, (0, 0, 0, 100))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=5))
    shadow_position = (position[0] + 5, position[1] + 5)  # Offset shadow for a natural effect
    background.paste(shadow_layer, shadow_position, shadow_layer)
    return background

def combine_images(background, obj_img):
    # Resize the object to fit on the background
    scaling_factor = background.width // 4
    obj_img.thumbnail((scaling_factor, scaling_factor), Image.Resampling.LANCZOS)
    
    # Set the position to place the object on the background
    pos_y = int(background.height * 0.6)  # Vertical placement
    pos_x = int(background.width * 0.7)   # Horizontal placement
    
    # Adjust object appearance to match the background
    obj_img = adjust_image(obj_img, background)
    
    # Add shadow effect behind the object
    background = add_shadow_effect(background, obj_img, (pos_x, pos_y))
    
    # Paste the object onto the background
    background.paste(obj_img, (pos_x, pos_y), obj_img)
    
    return background

def create_image(image_path, prompt, save_path):
    # Load object image and remove its background
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        obj_img_data = rembg.remove(img_data)
        obj_img = Image.open(io.BytesIO(obj_img_data)).convert("RGBA")

    # Initialize the Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the background based on the text prompt
    background_img = pipeline(prompt, height=512, width=768).images[0].convert("RGBA")

    # Combine the object image with the generated background
    final_img = combine_images(background_img, obj_img)

    # Save the resulting image
    final_img.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a background image from a prompt and blend it with an object.")
    parser.add_argument("--image", required=True, help="Path to the object image to blend.")
    parser.add_argument("--text-prompt", required=True, help="Text prompt for generating the background.")
    parser.add_argument("--output", required=True, help="Path to save the final blended image.")

    args = parser.parse_args()

    create_image(args.image, args.text_prompt, args.output)
