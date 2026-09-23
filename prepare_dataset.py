import os
import sys
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def prepare_merged_dataset(
    base_dir="dataset",
    output_dir="dataset/merged_dataset",
    max_uavdt_val_samples=3000
):
    print("=" * 60)
    print("STARTING DATASET PREPARATION & CONSOLIDATION")
    print("=" * 60)
    
    base_path = Path(base_dir).resolve()
    out_path = Path(output_dir).resolve()
    
    img_train_dir = out_path / "images" / "train"
    img_val_dir = out_path / "images" / "val"
    lbl_train_dir = out_path / "labels" / "train"
    lbl_val_dir = out_path / "labels" / "val"
    
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    visdrone_class_map = {
        0: 0,  # pedestrian -> person
        1: 0,  # people -> person
        3: 1,  # car -> vehicle
        4: 1,  # van -> vehicle
        5: 1,  # truck -> vehicle
        8: 1,  # bus -> vehicle
    }
    
    uavdt_class_map = {
        "car": 1,
        "van": 1,
        "truck": 1,
        "bus": 1,
        "vehicle": 1,
        "person": 0,
        "pedestrian": 0,
        "people": 0
    }
    
    # -----------------------------------------------------------------
    # 1. PROCESS VISDRONE DATASET
    # -----------------------------------------------------------------
    print("\n[1/2] Processing VisDrone Dataset...")
    visdrone_dir = base_path / "VisDrone"
    
    for split in ["train", "val"]:
        v_img_dir = visdrone_dir / "images" / split
        v_lbl_dir = visdrone_dir / "labels" / split
        
        target_img_dir = img_train_dir if split == "train" else img_val_dir
        target_lbl_dir = lbl_train_dir if split == "train" else lbl_val_dir
        
        if not v_img_dir.exists() or not v_lbl_dir.exists():
            print(f"Skipping VisDrone split '{split}', directory not found.")
            continue
            
        lbl_files = list(v_lbl_dir.glob("*.txt"))
        print(f"Processing VisDrone '{split}' set ({len(lbl_files)} files)...")
        
        processed_count = 0
        for lbl_path in tqdm(lbl_files, desc=f"VisDrone {split}"):
            img_path = v_img_dir / f"{lbl_path.stem}.jpg"
            if not img_path.exists():
                # Check for png or jpeg
                for ext in [".png", ".jpeg"]:
                    alt = v_img_dir / f"{lbl_path.stem}{ext}"
                    if alt.exists():
                        img_path = alt
                        break
            if not img_path.exists():
                continue
                
            # Filter and convert label
            new_lines = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            if cid in visdrone_class_map:
                                new_cid = visdrone_class_map[cid]
                                new_lines.append(f"{new_cid} {' '.join(parts[1:5])}\n")
                        except ValueError:
                            continue
            
            if new_lines:
                dest_img_name = f"visdrone_{img_path.name}"
                dest_lbl_name = f"visdrone_{lbl_path.stem}.txt"
                
                dest_img_path = target_img_dir / dest_img_name
                dest_lbl_path = target_lbl_dir / dest_lbl_name
                
                shutil.copyfile(img_path, dest_img_path)
                with open(dest_lbl_path, "w") as f:
                    f.writelines(new_lines)
                processed_count += 1
                
        print(f"VisDrone {split}: Successfully exported {processed_count} images & labels.")

    # -----------------------------------------------------------------
    # 2. PROCESS UAVDT DATASET
    # -----------------------------------------------------------------
    print("\n[2/2] Processing UAVDT Dataset...")
    uavdt_dir = base_path / "UAVDT"
    
    # Process train split -> target train
    u_train_img_dir = uavdt_dir / "train" / "img"
    u_train_ann_dir = uavdt_dir / "train" / "ann"
    
    if u_train_img_dir.exists() and u_train_ann_dir.exists():
        ann_files = list(u_train_ann_dir.glob("*.json"))
        print(f"Processing UAVDT 'train' set ({len(ann_files)} files)...")
        
        processed_count = 0
        for ann_path in tqdm(ann_files, desc="UAVDT train"):
            if ann_path.name == "meta.json":
                continue
            
            img_name = ann_path.stem  # e.g. M0101_img000001.jpg
            img_path = u_train_img_dir / img_name
            if not img_path.exists():
                continue
                
            with open(ann_path, "r") as f:
                data = json.load(f)
                
            size = data.get("size", {})
            img_w = size.get("width")
            img_h = size.get("height")
            
            if not img_w or not img_h:
                continue
                
            new_lines = []
            for obj in data.get("objects", []):
                cls_title = str(obj.get("classTitle", "")).lower()
                if cls_title in uavdt_class_map:
                    target_cid = uavdt_class_map[cls_title]
                    pts = obj.get("points", {}).get("exterior", [])
                    if len(pts) == 2:
                        x1, y1 = pts[0]
                        x2, y2 = pts[1]
                        
                        x1 = max(0, min(img_w, x1))
                        x2 = max(0, min(img_w, x2))
                        y1 = max(0, min(img_h, y1))
                        y2 = max(0, min(img_h, y2))
                        
                        bw = x2 - x1
                        bh = y2 - y1
                        if bw <= 0 or bh <= 0:
                            continue
                            
                        xc = (x1 + x2) / 2.0 / img_w
                        yc = (y1 + y2) / 2.0 / img_h
                        w_norm = bw / img_w
                        h_norm = bh / img_h
                        
                        new_lines.append(f"{target_cid} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            if new_lines:
                dest_img_name = f"uavdt_{img_path.name}"
                dest_lbl_name = f"uavdt_{ann_path.stem}.txt"
                
                dest_img_path = img_train_dir / dest_img_name
                dest_lbl_path = lbl_train_dir / dest_lbl_name
                
                shutil.copyfile(img_path, dest_img_path)
                with open(dest_lbl_path, "w") as f:
                    f.writelines(new_lines)
                processed_count += 1
                
        print(f"UAVDT train: Successfully exported {processed_count} images & labels.")

    # Process test split -> sample for target val
    u_test_img_dir = uavdt_dir / "test" / "img"
    u_test_ann_dir = uavdt_dir / "test" / "ann"
    
    if u_test_img_dir.exists() and u_test_ann_dir.exists():
        ann_files = list(u_test_ann_dir.glob("*.json"))
        # Sample evenly across sequence files to get representative validation set
        if len(ann_files) > max_uavdt_val_samples:
            step = len(ann_files) // max_uavdt_val_samples
            ann_files = ann_files[::step][:max_uavdt_val_samples]
            
        print(f"Processing UAVDT 'test/val' set ({len(ann_files)} sampled files)...")
        
        processed_count = 0
        for ann_path in tqdm(ann_files, desc="UAVDT val"):
            if ann_path.name == "meta.json":
                continue
                
            img_name = ann_path.stem
            img_path = u_test_img_dir / img_name
            if not img_path.exists():
                continue
                
            with open(ann_path, "r") as f:
                data = json.load(f)
                
            size = data.get("size", {})
            img_w = size.get("width")
            img_h = size.get("height")
            
            if not img_w or not img_h:
                continue
                
            new_lines = []
            for obj in data.get("objects", []):
                cls_title = str(obj.get("classTitle", "")).lower()
                if cls_title in uavdt_class_map:
                    target_cid = uavdt_class_map[cls_title]
                    pts = obj.get("points", {}).get("exterior", [])
                    if len(pts) == 2:
                        x1, y1 = pts[0]
                        x2, y2 = pts[1]
                        
                        x1 = max(0, min(img_w, x1))
                        x2 = max(0, min(img_w, x2))
                        y1 = max(0, min(img_h, y1))
                        y2 = max(0, min(img_h, y2))
                        
                        bw = x2 - x1
                        bh = y2 - y1
                        if bw <= 0 or bh <= 0:
                            continue
                            
                        xc = (x1 + x2) / 2.0 / img_w
                        yc = (y1 + y2) / 2.0 / img_h
                        w_norm = bw / img_w
                        h_norm = bh / img_h
                        
                        new_lines.append(f"{target_cid} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            if new_lines:
                dest_img_name = f"uavdt_{img_path.name}"
                dest_lbl_name = f"uavdt_{ann_path.stem}.txt"
                
                dest_img_path = img_val_dir / dest_img_name
                dest_lbl_path = lbl_val_dir / dest_lbl_name
                
                shutil.copyfile(img_path, dest_img_path)
                with open(dest_lbl_path, "w") as f:
                    f.writelines(new_lines)
                processed_count += 1
                
        print(f"UAVDT val: Successfully exported {processed_count} images & labels.")

    # -----------------------------------------------------------------
    # 3. CREATE DATASET YAML
    # -----------------------------------------------------------------
    yaml_path = out_path / "merged_data.yaml"
    abs_out_path = out_path.resolve().as_posix()
    
    yaml_content = f"""path: {abs_out_path}
train: images/train
val: images/val

names:
  0: person
  1: vehicle
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"\nCreated dataset YAML file at: {yaml_path}")
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print(f"Train images: {len(list(img_train_dir.glob('*')))}")
    print(f"Val images:   {len(list(img_val_dir.glob('*')))}")
    print("=" * 60)

if __name__ == "__main__":
    prepare_merged_dataset()
