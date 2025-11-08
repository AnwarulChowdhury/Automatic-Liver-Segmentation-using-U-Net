
# ======================
# 1. Imports
# ======================
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ======================
# 2. U-Net Definition 
# ======================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))
        return self.outc(x)

# ======================
# 3. Load Data
# ======================
def load_images_and_masks(images_dir, masks_dir, image_exts=('.jpg', '.png', '.jpeg')):
    images = []
    masks = []
    
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(image_exts) and not filename.lower().endswith('_mask.png'):
            img_path = os.path.join(images_dir, filename)
            
            # Grayscale image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image, axis=-1)

            mask_name = os.path.splitext(filename)[0] + '_mask.png'
            mask_path = os.path.join(masks_dir, mask_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # ✅ Convert 255 → 1
                mask = (mask > 127).astype(np.uint8)

                images.append(image)
                masks.append(mask)
            else:
                print(f"Warning: No mask found for {filename}")
    
    return np.array(images), np.array(masks)

# Directories
images_dir = 'C:/Users/anwarcho/Downloads/Liver_Medical_Image _Datasets/Liver_Medical_Image _Datasets/Images'
masks_dir = 'C:/Users/anwarcho/Downloads/Liver_Medical_Image _Datasets/Liver_Medical_Image _Datasets/Labels'

images, masks = load_images_and_masks(images_dir, masks_dir)

# First, separate 5 image for testing
remaining_imgs, test_imgs, remaining_masks, test_masks = train_test_split(
    images, masks, test_size=5, random_state=42
)

# Then split the remaining into 360 training and 39 validation
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    remaining_imgs, remaining_masks, train_size=360, test_size=35, random_state=42
)

print(f"Train: {train_imgs.shape}, Val: {val_imgs.shape}, Test: {test_imgs.shape}")


# Batch Generator
def get_batches(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size], y[i:i+batch_size]

# ======================
# 4. Train U-Net
# ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=1, n_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 20
batch_size = 4

train_losses = []
val_losses = []

best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in get_batches(train_imgs, train_masks, batch_size):
        x_batch = torch.tensor(x_batch, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / (len(train_imgs) / batch_size)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in get_batches(val_imgs, val_masks, batch_size):
            x_batch = torch.tensor(x_batch, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    val_loss /= (len(val_imgs) / batch_size)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_unet.pth")
        print("Saved new best model!")

# ======================
# Plot Loss Curve
# ======================
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_per_epoch.png")


# ======================
# 5. Evaluate & Metrics
# ======================
def compute_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    intersection = np.logical_and(pred == 1, target == 1).sum()
    union = np.logical_or(pred == 1, target == 1).sum()
    iou = intersection / (union + 1e-6)
    acc = (pred == target).sum() / len(target)
    return iou, acc

model.load_state_dict(torch.load("best_unet.pth"))
model.eval()

ious, accuracies = [], []
with torch.no_grad():
    for x_batch, y_batch in get_batches(val_imgs, val_masks, batch_size):
        x_batch = torch.tensor(x_batch, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        y_batch = torch.tensor(y_batch, dtype=torch.long)
        outputs = model(x_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for p, t in zip(preds, y_batch.numpy()):
            iou, acc = compute_metrics(p, t)
            ious.append(iou)
            accuracies.append(acc)

print("Mean IoU:", np.mean(ious))
print("Mean Pixel Accuracy:", np.mean(accuracies))


# ======================
#6. Single Image Segmentation & Save
# ======================


# Load the best model
model.load_state_dict(torch.load("best_unet.pth"))
model.eval()

# Select a single image from validation set
single_image = test_imgs[0]  # Shape: [H, W, 1]

# Preprocess
input_tensor = torch.tensor(single_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# Convert to binary mask image
segmented_image = (pred_mask * 255).astype(np.uint8)

# Save segmented image
cv2.imwrite("segmented_liver.png", segmented_image)
print("Segmented liver image saved as 'segmented_liver.png'")



# ======================
# Evaluate on First 3 Test Images
# ======================

def compute_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    intersection = np.logical_and(pred == 1, target == 1).sum()
    union = np.logical_or(pred == 1, target == 1).sum()
    iou = intersection / (union + 1e-6)
    acc = (pred == target).sum() / len(target)
    return iou, acc

ious, accuracies = [], []

model.eval()
with torch.no_grad():
    for idx in range(3): 
        image = test_imgs[idx]
        mask = test_masks[idx]

        # Prepare input
        input_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # Predict
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Compute IoU and Pixel Accuracy
        iou, acc = compute_metrics(pred_mask, mask)
        ious.append(iou)
        accuracies.append(acc)

        print(f"Image {idx+1}: IoU = {iou:.6f}, Pixel Accuracy = {acc:.6f}")

# Print overall mean values
print("\n===== Performance for First 3 Test Images =====")
print(f"Mean IoU: {np.mean(ious):.6f}")
print(f"Mean Pixel Accuracy: {np.mean(accuracies):.6f}")