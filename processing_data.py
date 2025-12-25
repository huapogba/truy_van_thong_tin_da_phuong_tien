import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import open_clip


# ===============================
# Dataset
# ===============================
class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx]


# ===============================
# Main
# ===============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Device: {device}")

    # ===============================
    # Load ViT-H-14 (MetaCLIP)
    # ===============================
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-H-14",
        pretrained="metaclip_fullcc",
        device=device
    )
    model.eval()

    # ===============================
    # DataLoader config
    # ===============================
    batch_size = 16          # H-14 rất nặng
    num_workers = 4          # Windows nên <= 4
    base_dir = "."           # thư mục chứa các folder ảnh

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith("L19_"):
            print(f"\n🔍 Đang xử lý thư mục: {folder_name}")

            dataset = ImageFolderDataset(folder_path, preprocess)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            all_features = []
            all_paths = []

            for images, paths in tqdm(loader):
                images = images.to(device, non_blocking=True)

                with torch.no_grad():
                    features = model.encode_image(images)
                    features = features / features.norm(dim=-1, keepdim=True)

                all_features.append(features.cpu().numpy())
                all_paths.extend(paths)

            # ===============================
            # Save results
            # ===============================
            if all_features:
                all_features = np.vstack(all_features)
                np.save(f"{folder_name}_features.npy", all_features)
                np.save(f"{folder_name}_paths.npy", np.array(all_paths))
                print(f"✅ Hoàn tất {len(all_paths)} ảnh từ {folder_name}")
            else:
                print(f"⚠️ Không có ảnh hợp lệ trong {folder_name}")


# ===============================
# Windows multiprocessing FIX
# ===============================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
