# Object in Scene Generator

## Problem Statement
Recent advancements in generative AI have led to the development of various creative workflows. One such workflow is utilizing generative AI techniques to create realistic product photographs, which are traditionally captured in studios, for display on e-commerce websites. The challenge at hand is to take an image of an object with a plain white background and generate a text-conditioned scene where the image of the object is placed naturally, resulting in a final output that appears coherent and realistic.

The perception of "realism" in the generated scene can vary based on several factors, including:
- Aspect ratio of the object relative to the scene
- Spatial placement of the object
- Lighting conditions in the scene that match the object's characteristics

The objective is to develop approaches that enhance the natural appearance of the scene.

## Approach
This project consists of two main tasks:

## 1. Image Generation
The first task is to write executable code that accepts the location of an image and a text prompt via command line arguments, generating an output image based on the specified parameters.

**Example command:**
```bash
python run.py --image ./example.jpg --text-prompt "product in a kitchen used in meal preparation" --output ./generated.png
```
### Key Components:
- Image Processing Libraries: The project uses Python libraries such as PIL for image processing, torch for model handling, and rembg for background removal.
- Stable Diffusion Model: Leveraging the StableDiffusionPipeline from Hugging Face’s diffusers library to generate backgrounds based on the text prompt.
## 2. Video Generation
Once an initial solution is in place, the next step is to create a small video output by generating multiple consistent frames. This can be accomplished by implementing a camera zoom-out effect or translating the scene in various directions.

### Key Components:
- Frame Generation: Generate a sequence of frames using a zoom factor or random translations.
- Video Creation: Utilize OpenCV to compile the generated frames into a video.

## Installation
- To run this project, ensure you have the following dependencies installed:
```bash
pip install torch torchvision torchaudio diffusers rembg Pillow opencv-python tqdm
```
## Usage
### Image Generation:
  - Run the command to generate an image:
    ```bash
    python run.py --image <path_to_object_image> --text-prompt "<your_text_prompt>" --output <output_image_path>
    ```
### Video Generation:
  - To create a video with zoom-out effects, use the following command:
     ```bash
    python run_video.py --image <path_to_object_image> --text-prompt "<your_text_prompt>" --output <output_video_path> --frames <number_of_frames>
    ```

## Results
### Example Outputs
- Input Image
- Generated Image
### Visual Results from Experiments
- Successful Experiment
- Failed Experiment

## Challenges Faced
- Achieving realism in terms of lighting and shadows was a significant challenge.
- The initial alignment of object placement often resulted in unnatural appearances, which required fine-tuning.

## Future Work
- Explore additional techniques for improving realism, such as implementing advanced shadowing and lighting effects.
- Experiment with different text prompts and objects to evaluate the versatility of the generated scenes.
