import os
import subprocess

def find_ts_files(directory):
    ts_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ts'):
                ts_files.append(os.path.join(root, file))
    return ts_files

def convert_ts_to_mp4(ts_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output .mp4 file name
    file_name = os.path.basename(ts_file).rsplit('.', 1)[0]
    output_file = os.path.join(output_dir, f"{file_name}.ts")
    
    command = [
        "ffmpeg",
        #"-hwaccel", "cuda",    # Uncomment for GPU acceleration
        "-i", ts_file,         # Input .ts file
        "-vf", "crop=640:640", # Video filter to crop to 640x640
        "-c:v", "libx264",     # Video codec (H.264)
        "-preset", "fast",     # Speed/quality tradeoff preset
        "-crf", "23",          # Constant Rate Factor (quality setting)
        output_file            # Output .mp4 file
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Conversion complete for: {ts_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion of {ts_file}: {e}")

def process_all_ts_files(directory, output_dir):
    ts_files = find_ts_files(directory)
    for ts_file in ts_files:
        convert_ts_to_mp4(ts_file, output_dir)

# Example usage:
directory_path = "D:/DemoVids"
output_dir = "D:/DemoVids/Output"
process_all_ts_files(directory_path, output_dir)
print("Done with all")
