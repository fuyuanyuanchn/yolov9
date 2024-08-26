import os
import shutil


def organize_files(source_folder):
    jpg_folder = os.path.join(source_folder, "images")
    txt_folder = os.path.join(source_folder, "labels")

    os.makedirs(jpg_folder, exist_ok=True)
    os.makedirs(txt_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        if os.path.isfile(file_path):
            if filename.lower().endswith('.jpg'):
                shutil.move(file_path, os.path.join(jpg_folder, filename))
            elif filename.lower().endswith('.txt'):
                shutil.move(file_path, os.path.join(txt_folder, filename))


# 使用示例
source_folder = "/path/to/your/source/folder"
organize_files(source_folder)
print("complete")