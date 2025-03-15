import os
import random
import csv
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import argparse
import glob

def list_digit_images(digit_folder):
    """
    Returns a list of all image paths in the specified digit folder.
    """
    # Get all PNG files in the digit folder
    image_paths = glob.glob(os.path.join(digit_folder, "*.png"))
    return image_paths

def create_digit_image_map(dataset_root):
    """
    Creates a dictionary mapping each digit to a list of its image paths.
    """
    digit_image_map = {}
    for digit in range(10):
        digit_folder = os.path.join(dataset_root, str(digit))
        if os.path.exists(digit_folder):
            digit_image_map[digit] = list_digit_images(digit_folder)
            print(f"Found {len(digit_image_map[digit])} images for digit {digit}")
        else:
            print(f"Warning: Folder for digit {digit} not found at {digit_folder}")
            digit_image_map[digit] = []
    
    return digit_image_map

def create_synthetic_sequence(digit_image_map, sequence_length=13, target_height=64, spacing=30, padding=15, random_scaling=True):
    """
    Creates a synthetic digit sequence image by concatenating random digit images.
    
    Args:
        digit_image_map: Dictionary mapping each digit to a list of its image paths
        sequence_length: Length of the sequence to generate (default: 13)
        target_height: Height to resize each digit image to (default: 64 pixels)
        spacing: Horizontal spacing between digits (default: 4 pixels)
        padding: Padding around the entire sequence (default: 8 pixels)
        random_scaling: Whether to apply slight random scaling to digits (default: True)
        
    Returns:
        tuple: (PIL Image of the sequence, string label of the sequence)
    """
    # remove 10% of spacing
    #spacing = int(spacing * random.uniform(0.95, 1.0))

    # Generate a random sequence of digits
    label = ''.join([str(random.randint(0, 9)) for _ in range(sequence_length)])
    
    # Select a random image for each digit in the sequence
    digit_images = []
    for digit_char in label:
        digit = int(digit_char)
        if not digit_image_map[digit]:
            print(f"Warning: No images available for digit {digit}. Replacing with random digit.")
            # Find a digit that has images
            available_digits = [d for d in digit_image_map.keys() if digit_image_map[d]]
            if not available_digits:
                raise ValueError("No digit images available in any digit folder")
            digit = random.choice(available_digits)
            label = label.replace(digit_char, str(digit), 1)
        
        # Select a random image for this digit
        image_path = random.choice(digit_image_map[digit])
        digit_images.append(image_path)
    
    # Open and resize each image
    resized_digits = []
    for image_path in digit_images:
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        
        # Get aspect ratio and calculate new width
        aspect_ratio = img.width / img.height
        
        # Apply slight random scaling if enabled
        height_variation = 0
        if random_scaling:
            # Random height variation of ±10%
            height_variation = random.uniform(-0.10, 0.10)
        
        new_height = int(target_height * (1 - height_variation))
        new_width = int(new_height * aspect_ratio)
        
        # Resize the image
        #img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        # resize image times 3
        #img = img.resize((int(img.width * 3), int(img.height * 3)), Image.LANCZOS)
        resized_digits.append(img)
    
    # Calculate total width needed
    total_width = 1175 #sum(img.width for img in resized_digits) + spacing * (len(resized_digits) - 1) + padding * 2
    
    # Calculate maximum height among the resized digits
    max_height = 140#max(img.height for img in resized_digits) + padding * 2
    
    # Create a blank image for the sequence
    sequence_img = Image.new('L', (total_width, max_height), color=255)
    
    # Paste each digit onto the sequence image
    x_offset = padding
    for img in resized_digits:
        # Calculate vertical position (center in the image)
        y_offset = padding + (max_height - padding * 2 - img.height) // 2
        sequence_img.paste(img, (x_offset, y_offset))
        x_offset += img.width + spacing
        #spacing = int(spacing * random.uniform(0.8, 0.9))
    
    # Convert to RGB
    sequence_img = sequence_img.convert("RGB")
    
    return sequence_img, label

def generate_dataset(dataset_root, output_dir, num_samples=1000, sequence_length=13):
    """
    Generates a synthetic dataset of digit sequences.
    
    Args:
        dataset_root: Path to the root directory containing digit folders (0-9)
        output_dir: Path to the directory where generated images will be saved
        num_samples: Number of synthetic samples to generate
        sequence_length: Length of each digit sequence
    """
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create digit image map
    digit_image_map = create_digit_image_map(dataset_root)
    
    # Create CSV file for labels
    csv_path = os.path.join(output_dir, "labels.csv")
    records = []
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_filename', 'label'])
        
        # Generate samples
        for i in tqdm(range(num_samples), desc="Generating synthetic samples"):
            # Generate synthetic sequence
            sequence_img, label = create_synthetic_sequence(digit_image_map, sequence_length)
            
            # Save the image
            image_filename = f"synthetic_{i:05d}.png"
            image_path = os.path.join(images_dir, image_filename)
            sequence_img.save(image_path)
            records.append({
                "image": image_filename,
                "label": label
            })
            # Write to CSV
            writer.writerow([image_filename, label])
    import json
    with open(os.path.join(output_dir, "labels.json"), 'w') as f:
        json.dump(records, f, indent=4)
    print(f"Dataset generation complete. {num_samples} samples generated.")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic digit sequence dataset")
    parser.add_argument("--dataset_root", type=str, default="/Users/erick/git/prod/account_docket_no/data/segmented_digits",
                        help="Path to the root directory containing digit folders (0-9)")
    parser.add_argument("--output_dir", type=str, default="data/synthetic",
                        help="Path to the directory where generated data will be saved")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--sequence_length", type=int, default=13,
                        help="Length of each digit sequence")
    
    args = parser.parse_args()
    
    # Generate dataset
    generate_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sequence_length=args.sequence_length
    )

if __name__ == "__main__":
    main() 