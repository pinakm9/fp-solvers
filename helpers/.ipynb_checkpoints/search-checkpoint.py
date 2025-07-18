import os

def search_string_in_files(folder_path, search_str, context=2, file_extensions=None):
    if file_extensions:
        file_extensions = tuple(file_extensions)

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith('.'):
                continue  # Skip hidden files
            if file_extensions and not file.endswith(file_extensions):
                continue  # Skip unwanted extensions

            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except (UnicodeDecodeError, FileNotFoundError):
                continue  # Skip unreadable files

            for i, line in enumerate(lines):
                if search_str in line:
                    print(f"\n🔍 Match in {file_path} at line {i+1}:")
                    start = max(0, i - context)
                    end = min(len(lines), i + context + 1)
                    for j in range(start, end):
                        prefix = '➡' if j == i else '  '
                        print(f"{prefix} {j+1:>4}: {lines[j].rstrip()}")
