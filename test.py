import os
import shutil
import random


def organize_files(source_folder, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
    assert train_ratio + valid_ratio + test_ratio == 1.0

    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(source_folder, split, subdir), exist_ok=True)

    all_files = sorted(os.listdir(source_folder))

    image_files = [f for f in all_files if f.endswith('.jpg')]
    label_files = [f for f in all_files if f.endswith('.txt')]

    assert len(image_files) == len(label_files), "图像和标签文件数量不匹配"

    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    valid_count = int(total_files * valid_ratio)
    test_count = total_files - train_count - valid_count

    for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
        if i < train_count:
            split = 'train'
        elif i < train_count + valid_count:
            split = 'valid'
        else:
            split = 'test'

        shutil.move(os.path.join(source_folder, image_file),
                    os.path.join(source_folder, split, 'images', image_file))

        shutil.move(os.path.join(source_folder, label_file),
                    os.path.join(source_folder, split, 'labels', label_file))

    print(f"文件整理完成。分配比例：训练集 {train_count}, 验证集 {valid_count}, 测试集 {test_count}")


if __name__ == "__main__":
    source_folder = "obj_train_data"
    organize_files(source_folder)