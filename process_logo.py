from PIL import Image, ImageOps
import os

def process_logo(input_path, output_dir):
    # Open the image
    img = Image.open(input_path)
    img = img.convert("RGBA")
    
    # Use get_flattened_data for newer Pillow versions if preferred, 
    # but getdata is still widely available despite deprecation warnings.
    datas = img.getdata()
    
    newData = []
    # Tolerance for "white"
    tolerance = 240 
    
    for item in datas:
        # If the pixel is very light (white-ish), make it transparent
        if item[0] > tolerance and item[1] > tolerance and item[2] > tolerance:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
            
    img.putdata(newData)
    
    # Crop to content
    # Get bounding box of non-zero alpha
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # Ensure output dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save original cropped
    img.save(os.path.join(output_dir, "logo.png"), "PNG")
    
    # Save resized versions
    sizes = [128, 256, 512]
    for size in sizes:
        # Use thumbnail to maintain aspect ratio
        resized = img.copy()
        resized.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # Create a new square transparent background
        new_img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        # Center the resized logo
        paste_pos = ((size - resized.width) // 2, (size - resized.height) // 2)
        new_img.paste(resized, paste_pos)
        
        new_img.save(os.path.join(output_dir, f"logo_{size}.png"), "PNG")
    
    print(f"Processed logo saved to {output_dir}")

if __name__ == "__main__":
    input_file = "images/duwhal.jpg"
    output_folder = "images"
    process_logo(input_file, output_folder)
