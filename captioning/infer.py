from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  #Preprocessing tool for images
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device) #Caption generator model
model.eval()

def generate_caption(image_path, max_length=30, num_beams=4):
    assert os.path.isfile(image_path), f"File not found: {image_path}"
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

    return processor.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    img_path = r"C:\Users\suyas\OneDrive\Desktop\image_caption_segmentation_project\data\val2017\000000000885.jpg"
    caption = generate_caption(img_path)
    print("Caption:", caption)
    
    