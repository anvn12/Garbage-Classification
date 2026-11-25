import random
import shutil
from pathlib import Path


def copy_files(file_list, target_dir):
    """Copy a list of image files into the destination folder."""
    # Create folder if it does not exist
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in file_list:
        # Copy image to destination
        shutil.copy2(file_path, target_dir / file_path.name)


def split_dataset(
    src_dir, dst_root, class_name, train_ratio=0.8, val_ratio=0.1, seed=42
):
    """Split images from one class folder into train, val, and test folders."""
    random.seed(seed)  # Fix seed so the split is reproducible
    valid_exts = {".jpg", ".jpeg", ".png"}

    # Collect and shuffle all valid images
    images = sorted(
        p
        for p in Path(src_dir).iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    )

    random.shuffle(images)

    # Slice list into train (80%), val (10%), and test (10%)
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = images[:n_train]
    val_files = images[n_train : n_train + n_val]
    test_files = images[n_train + n_val :]

    # Copy each split into dataset/<split>/<class_name>
    out_root = Path(dst_root)
    copy_files(train_files, out_root / "train" / class_name)
    copy_files(val_files, out_root / "val" / class_name)
    copy_files(test_files, out_root / "test" / class_name)

    print(
        f"[{class_name}] Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}"
    )


if __name__ == "__main__":
    # train-val-test: 0.8-0.1-0.1; seed=42
    original_dataset_dir = "./Garbage-Classification"

    split_dataset(
        src_dir=original_dataset_dir + "/cardboard",
        dst_root="dataset",
        class_name="cardboard",
    )
    split_dataset(
        src_dir=original_dataset_dir + "/glass",
        dst_root="dataset",
        class_name="glass",
    )
    split_dataset(
        src_dir=original_dataset_dir + "/metal",
        dst_root="dataset",
        class_name="metal",
    )
    split_dataset(
        src_dir=original_dataset_dir + "/paper",
        dst_root="dataset",
        class_name="paper",
    )
    split_dataset(
        src_dir=original_dataset_dir + "/plastic",
        dst_root="dataset",
        class_name="plastic",
    )
