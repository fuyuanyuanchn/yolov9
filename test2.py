import os

def update_txt_file(txt_file_path, images_folder):
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    if len(image_files) != len(lines):
        print(f"警告：图片数量 ({len(image_files)}) 与txt文件行数 ({len(lines)}) 不匹配！")
        return

    new_txt_file_path = txt_file_path.replace('.txt', '_updated.txt')

    with open(new_txt_file_path, 'w') as new_file:
        for image_file, line in zip(image_files, lines):
            updated_line = line.replace('night_output.mp4', image_file)
            new_file.write(updated_line)

    print(f"new txt file path: {new_txt_file_path}")

txt_file_path = 'path/to/your/original_file.txt'  # 替换为实际的txt文件路径
images_folder = 'path/to/your/images_folder'  # 替换为实际的图片文件夹路径

update_txt_file(txt_file_path, images_folder)