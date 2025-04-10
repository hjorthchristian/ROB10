from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lang_sam import LangSAM

# Load model and image
model = LangSAM(sam_type="sam2.1_hiera_tiny")
image_pil = Image.open("/home/chrishj/Pictures/Screenshots/Screenshot from 2025-04-10 09-55-17.png").convert("RGB")
text_prompt = "pallet"
results = model.predict([image_pil], [text_prompt])
data = results[0]  # assuming results is a list with one dict

# Debug information
print("Data keys:", data.keys())
for key in data:
    print(f"{key}: type={type(data[key])}")
    if isinstance(data[key], np.ndarray):
        print(f"  shape={data[key].shape}")

# Convert PIL image to NumPy array
image = np.array(image_pil)

# Copy for drawing
masked_image = image.copy()

# Visualize detections
for i in range(len(data['labels'])):
    mask = data['masks'][i]
    label = data['labels'][i]
    box = data['boxes'][i]
    score = data['scores'][i]
    
    # Handle mask_scores - checking specifically for ndim > 0
    if isinstance(data['mask_scores'], np.ndarray) and data['mask_scores'].ndim > 0:
        mask_score = data['mask_scores'][i] if i < data['mask_scores'].shape[0] else float(data['mask_scores'])
    else:
        # It's a scalar
        mask_score = float(data['mask_scores'])

    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    colored_mask = np.zeros_like(masked_image)
    for c in range(3):
        colored_mask[:, :, c] = (mask * color[c]).astype(np.uint8)

    # Blend mask with original image
    masked_image = cv2.addWeighted(masked_image, 1.0, colored_mask, 0.5, 0)

    # Draw bounding box
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(masked_image, (x1, y1), (x2, y2), color.tolist(), 2)

    # Put label and scores
    text = f"{label}: {score:.2f}, mask: {mask_score:.2f}"
    cv2.putText(masked_image, text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1, cv2.LINE_AA)

# Display result
plt.figure(figsize=(12, 8))
plt.imshow(masked_image)
plt.axis("off")
plt.title("Detected Objects with Masks")
plt.show()