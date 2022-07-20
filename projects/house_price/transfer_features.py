"""特徴量のパスを変えたので、引っ越しするためのスクリプト."""

import glob
import shutil
from pathlib import Path


def copy(from_dir, to_dir):
    train_file = from_dir / "train.ftr"
    test_file = from_dir / "test.ftr"
    shutil.copy(train_file, to_dir)
    shutil.copy(test_file, to_dir)


def copy_one_feat(feat_name):
    target_dir = "data/features/v5"
    target_dirs = glob.glob(f"{target_dir}/{feat_name}_*")
    dest = Path("data/features/v5_sub")
    for target in target_dirs:
        print(target)
        hash = target.split("_")[-1]
        target = Path(target)
        new_feat_path = dest / feat_name / hash
        new_feat_path.mkdir(parents=True, exist_ok=True)
        print(f"copy from {target} to {new_feat_path}")
        copy(target, new_feat_path)


feat_names = ["CategoriesLdaVectorizer", "_GroupFeatures"]


for feat_name in feat_names:
    copy_one_feat(feat_name=feat_name)
