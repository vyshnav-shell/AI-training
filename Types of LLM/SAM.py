from transformers import SamModel, SamProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt

def run_sam(image_path):

    image = Image.open(image_path).convert("RGB")

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    w, h = image.size

    inputs = processor(
        image,
        input_points=[[[w // 2, h // 2]]],
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    print("Mask Shape:", outputs.pred_masks.shape)
    print("IoU Scores:", outputs.iou_scores)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    plt.imshow(masks[0][0][0], cmap="gray")
    plt.title("Best Predicted Mask")
    plt.axis("off")
    plt.show()

run_sam("data_center.jpg")