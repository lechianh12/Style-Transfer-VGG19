from torchvision import transforms
from PIL import Image
import os
import torch
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3


import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def load_images_from_folder(folder_path, transform):
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')  # Hỗ trợ cả viết hoa
    for filename in os.listdir(folder_path):
        if filename.endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img))
    if not images:  # Nếu không có ảnh
        raise ValueError(f"No images found in folder: {folder_path}")
    return torch.stack(images)


transform = transforms.Compose([
    transforms.Resize((299, 299)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




def get_features(images, model):
    with torch.no_grad():
        features = model(images)
        print(f"Shape of features: {features.shape}")  # Debug: Kiểm tra kích thước đầu ra
        if len(features.shape) == 4:  # Nếu đầu ra có 4 chiều, áp dụng pooling
            features = adaptive_avg_pool2d(features, (1, 1))
        return features.view(features.size(0), -1)  # Trả về tensor 2D


model = inception_v3(pretrained=True, transform_input=False)
model.fc = torch.nn.Identity()  
model.eval()  

# Load ảnh
content_images = load_images_from_folder("E:/DIP project/Forest", transform)
output_images = load_images_from_folder("E:/DIP project/output_images_vgg19", transform)

print(content_images.shape)  # Xem kích thước của tensor
print(output_images.shape)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)  # Chuyển mô hình sang GPU
# content_images = content_images.to(device)
# output_images = output_images.to(device)


# Trích xuất đặc trưng
content_features = get_features(content_images, model)
output_features = get_features(output_images, model)

# Tính trung bình và hiệp phương sai
mu1, sigma1 = content_features.mean(0).numpy(), np.cov(content_features.numpy(), rowvar=False)
mu2, sigma2 = output_features.mean(0).numpy(), np.cov(output_features.numpy(), rowvar=False)

# Tính FID
fid = calculate_fid(mu1, sigma1, mu2, sigma2)
print(f"FID: {fid}")

