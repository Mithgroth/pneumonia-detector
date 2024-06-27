from fastai.vision.all import *
import shutil
import os
from pathlib import Path
import random

def setup_directories():
    path = Path("data/")
    normal_path = path / "NORMAL"
    pneumonia_path = path / "PNEUMONIA"
    test_path_normal = path / "test" / "NORMAL"
    test_path_pneumonia = path / "test" / "PNEUMONIA"
    
    for p in [test_path_normal, test_path_pneumonia]:
        p.mkdir(parents=True, exist_ok=True)
        
    return normal_path, pneumonia_path, test_path_normal, test_path_pneumonia

def split_data(source_path, test_path, split_ratio=0.1):
    all_images = get_image_files(source_path)
    random.shuffle(all_images)
    num_to_test = int(len(all_images) * split_ratio)
    
    test_images = all_images[:num_to_test]
    for img in test_images:
        shutil.move(img, test_path)
        
    print(f"Moved {len(test_images)} images to {test_path}")

def train():
    normal_path, pneumonia_path, test_path_normal, test_path_pneumonia = setup_directories()
    
    # Split the data 10% to test folders
    split_data(normal_path, test_path_normal, 0.1)
    split_data(pneumonia_path, test_path_pneumonia, 0.1)
    
    # Setup the DataBlock for training
    path = Path("data/")
    data = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),  # 20% data for validation
        get_y=parent_label,
        item_tfms=Resize(460),
        batch_tfms=aug_transforms(size=224)
    )

    dls = data.dataloaders(path)
    
    # Train the model
    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(1)
    
    # Export the model
    output_dir = Path("output/")
    output_dir.mkdir(parents=True, exist_ok=True)
    learn.export(output_dir / 'model.pkl')

if __name__ == "__train__":
    train()
