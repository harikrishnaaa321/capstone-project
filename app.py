import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from huggingface_hub import hf_hub_download
import os

# ===============================
# 1. LOAD MODEL FROM HF
# ===============================

model_path = hf_hub_download(
    repo_id="harikrishnaaa321/cnn_attention_model",
    filename="cnn_attention_best.pth"
)

# ===============================
# 2. MODEL ARCHITECTURE
# ===============================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.se(x)
        return x * w

class CNN_Attention_Model(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_Attention_Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(64),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(128)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===============================
# 3. LOAD MODEL
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Attention_Model(num_classes=4).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

labels = ["Glioma", "Meningioma", "Pituitary", "Normal"]

# Direct images (not folder)
example_images = {
    "Glioma": "./glioma.jpg",
    "Meningioma": "./meningioma.jpg",
    "Pituitary": "./pituitary.jpg",
    "Normal": "./notumor.jpg"
}

# ===============================
# 4. TRANSFORMS
# ===============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ===============================
# 5. PREDICT FUNCTION
# ===============================

def predict_tumor(image):
    # Convert to grayscale
    image = image.convert("L")

    # Make 2-channel input
    img_tensor = transform(image).repeat(2, 1, 1).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs)
    pred_label = labels[pred_idx]
    confidences = {labels[i]: float(probs[i]) for i in range(len(labels))}

    # ===============================
    # Create reference panel (CREAM BG)
    # ===============================

    example_imgs = [Image.open(example_images[label]).resize((224, 224)) for label in labels]

    combined_width = 224 * 4
    combined_height = 224 + 40

    combined_image = Image.new("RGB", (combined_width, combined_height), "#f8eecf")  # cream

    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.load_default()

    for i, img in enumerate(example_imgs):
        x = 224 * i
        combined_image.paste(img, (x, 0))

        # Highlight predicted class with red border
        if i == pred_idx:
            draw.rectangle([x, 0, x + 223, 223], outline="#ff4444", width=5)

        # Draw label text
        text = labels[i]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + (224 - tw) // 2
        draw.text((tx, 228), text, fill="black", font=font)

    return pred_label, confidences, combined_image


# ===============================
# 6. CREAM UI THEME CSS
# ===============================

custom_css = """
:root {
    --body-background-fill: #f8eecf;
    --block-background-fill: #fffaf0;
    --border-color: #d5c7a1;
    --button-primary-background-fill: #d6b77a;
    --button-primary-text-color: #000000;
}

body, .gradio-container {
    background: #f8eecf !important;
}

.gr-button {
    border-radius: 8px !important;
}
"""

# ===============================
# 7. GRADIO INTERFACE
# ===============================

interface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil", label="Upload MRI Scan"),
    outputs=[
        gr.Label(label="Predicted Tumor Type"),
        gr.Label(label="Confidence Scores"),
        gr.Image(label="Reference Tumor Images")
    ],
    title="🧠 Brain Tumor Classification (Attention CNN)",
    description="Upload an MRI scan. Model predicts tumor type and shows reference examples.",
    css=custom_css
)

if __name__ == "__main__":
    interface.launch()
