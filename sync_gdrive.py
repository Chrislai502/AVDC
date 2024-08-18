import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def list_gdrive_files(folder_id):
    result = subprocess.run(['gdrive', 'files', 'list', '--parent', folder_id], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    files = {}
    for line in output.split('\n')[1:]:
        parts = line.split()
        if len(parts) >= 6:
            file_id = parts[0]
            file_name = parts[1]
            if parts[2] == 'folder':
                file_date = ' '.join(parts[4:6])
            else:
                file_date = ' '.join(parts[5:7])
            files[file_name] = {'id': file_id, 'date': file_date}
    return files

def get_gdrive_folder_id(folder_name):
    folders = {
        "aloha": "1-VUtDa5rCysvLcrjLKtqOnrk-caqkotV",
        "avdc": "1bMYZZix7fjgP-rfCmDp3tRxV9pj6nOvk"
    }
    return folders.get(folder_name.lower())

def get_local_files(folder_path):
    local_files = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            file_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            local_files[file] = {'path': str(file_path), 'date': file_date}
    return local_files

def upload_file(folder_id, file_path):
    result = subprocess.run(['gdrive', 'files', 'upload', '--parent', folder_id, file_path], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

def sync_files(gdrive_files, local_files, folder_id):
    for local_file, local_info in local_files.items():
        if local_file not in gdrive_files or gdrive_files[local_file]['date'] != local_info['date']:
            print(local_file not in gdrive_files)
            if local_file in gdrive_files:
                print(gdrive_files[local_file]['date'])
                print(local_info['date'])
            print(f"Uploading {local_file}...")
            upload_file(folder_id, local_info['path'])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sync_gdrive.py <aloha|avdc> <local_folder_path>")
        sys.exit(1)

    folder_name = sys.argv[1]
    local_folder_path = sys.argv[2]

    folder_id = get_gdrive_folder_id(folder_name)
    if not folder_id:
        print("Invalid folder name. Choose 'aloha' or 'avdc'.")
        sys.exit(1)

    gdrive_files = list_gdrive_files(folder_id)
    local_files = get_local_files(local_folder_path)
    print("Gdrive: ", gdrive_files, "\n")
    print("Local: ", local_files, "\n")

    sync_files(gdrive_files, local_files, folder_id)
