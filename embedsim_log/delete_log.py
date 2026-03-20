import os

def delete_log_files():
    current_dir = os.getcwd()
    deleted_files = 0

    for filename in os.listdir(current_dir):
        if filename.endswith(".log"):
            file_path = os.path.join(current_dir, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
                deleted_files += 1
            except Exception as e:
                print(f"Error deleting {filename}: {e}")

    print(f"\nTotal .log files deleted: {deleted_files}")

delete_log_files()