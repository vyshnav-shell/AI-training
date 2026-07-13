from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def run_vlm(image_path):

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    print("Vision Language Model")
    print("---------------------")
    print("Generated Description:")
    print(caption)


run_vlm("data_center.jpg")