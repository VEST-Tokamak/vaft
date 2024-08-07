import subprocess
from tools.path_tools import create_file_path

def delete(shot, username, run):
    file_path = create_file_path(shot, username, run, new=False)

    try:
        result = subprocess.run(
            ["hsrm",  file_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"File {file_path} has been deleted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to delete file {file_path}. Error: {e.stderr.decode()}")
