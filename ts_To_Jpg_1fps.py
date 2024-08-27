import os
import subprocess

def find_ts_files(directory):
    ts_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ts'):
                ts_files.append(os.path.join(root, file))
    return ts_files

def convert_ts_to_jpgs(ts_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate a unique prefix for each file to avoid overwriting
    file_name = os.path.basename(ts_file).rsplit('.', 1)[0]
    output_pattern = os.path.join(output_dir, f"{file_name}_frame_%04d.jpg")
    
    command = [
        "ffmpeg",
        #"-hwaccel", "cuda",        
        #"-hwaccel_output_format", "cuda",
        "-i", ts_file,
        "-vf", "fps=1",            
        "-c:v", "mjpeg",          
        output_pattern
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Conversion complete for: {ts_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion of {ts_file}: {e}")

def process_all_ts_files(directory, output_dir):
    ts_files = find_ts_files(directory)
    for ts_file in ts_files:
        convert_ts_to_jpgs(ts_file, output_dir)

# Example usage:
directory_path = "C:/Git/ml/DataManagement/datasets/06All/06_TS_Vids"
output_dir = "C:/Git/ml/DataManagement/datasets/06All/06_JPGs"
process_all_ts_files(directory_path, output_dir)
print("Done with all")
