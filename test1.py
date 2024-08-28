import os

folder_path = 'your_folder_path_here'

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r') as file:
            lines = file.readlines()

        modified_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) > 0:
                if parts[0] == '0':
                    parts[0] = '1'
                elif parts[0] == '1':
                    parts[0] = '0'
            modified_lines.append(' '.join(parts) + '\n')

        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

        print(f"Modified: {filename}")

    elif filename.endswith('.jpg'):
        print(f"Skipped: {filename}")
