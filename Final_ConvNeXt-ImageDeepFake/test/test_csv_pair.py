import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.convnext import build_model
from transforms import test_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


TEST_CSV = r"C:\Users\touto\Desktop\test\kaggle\test.csv"
BASE_DIR = r"C:\Users\touto\Desktop\test\kaggle"  
OUT_CSV  = r"C:\Users\touto\Desktop\test\kaggle\submission.csv"
CKPT     = r"train\checkpoint1_phase3.pth"
BATCH_SIZE = 32
NUM_WORKERS = 10

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, CKPT)

    dataset = CSVDataset(
        csv_path=TEST_CSV,
        base_dir=BASE_DIR,
        transform=test_transforms()
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,              
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device, non_blocking=True)

            logits = model(images)

            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

            all_ids.extend(ids)
            all_preds.extend(preds)

    # Write submission with ONLY: id,label (same order as test.csv)
    sub = pd.DataFrame({"id": all_ids, "label": all_preds})
    sub.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print("Rows:", len(sub))
    print(sub.head())



class CSVDataset(Dataset):
    """
    Reads test.csv with column: id (relative path like test_data/xxx.jpg)
    Loads image from: base_dir / id
    Returns: (image_tensor, id_str)
    """
    def __init__(self, csv_path, base_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        if "id" not in self.df.columns:
            raise ValueError(f"CSV must contain an 'id' column. Found: {self.df.columns.tolist()}")

        self.ids = self.df["id"].astype(str).tolist()
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rel_path = self.ids[idx]
        img_path = os.path.join(self.base_dir, rel_path)

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, rel_path


def load_model(device, ckpt_path="train/checkpoint3.pth"):
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=device)

    # Support both: state_dict only OR dict checkpoint
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    main()
