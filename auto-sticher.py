# monitors RAW_DIR for new .insv files, when new files are found it checks if that file is still being written to,
# if not it will add to a list of files to be processed. When the list of files to be processed is not empty 
# it will process the files in the list.
# The script busy waits for SCAN_INTERVAL seconds before checking for new files again.
# The script will run indefinitely until killed.
# The format of the files is VID_YYYYMMDD_HHMMSS_[10]0_X.insv, the format of the processed files
# is VID_YYYYMMDD_HHMMSS.mp4
# The script expects MediaSDKTest in the PATH

import os
import time 
import subprocess
import re

RAW_DIR = "/media/raw"
OUT_DIR = "/media/stitched"
OUTPUT_SIZE = "5760x2880"
BITRATE = "200000000"
STITCH_TYPE = "dynamicstitch"
SCAN_INTERVAL = 600

def get_files():
    files = []
    for file in os.listdir(RAW_DIR):
        if file.endswith(".insv"):
            files.append(file)
    return files

def is_writing(file):
    # check file size, if it is increasing it is still being written to
    file_path = os.path.join(RAW_DIR, file)
    #check if file exists
    if not os.path.exists(file_path):
        return True
    size = os.path.getsize(file_path)
    time.sleep(1)
    new_size = os.path.getsize(file_path)
    return size != new_size

def process_file(file):
    file1, file2, output, timestamp = file
    print("Processing file: " + output)
    log    = os.path.join(OUT_DIR, "VID_" + timestamp + ".log")
    cmd = ["MediaSDKTest", "-inputs", file1, file2, "-output_size", OUTPUT_SIZE, "-bitrate" , BITRATE, \
                          "-stitch_type", STITCH_TYPE, "-output", output]
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=f)
    #print(cmd)
                          
    # os.rename(os.path.join(RAW_DIR, file), os.path.join(PROCESSING_DIR, file))
    print("Processed file: " + output)

def main():
    last_time_modified = 0
    while True:
        # check if RAW_DIR was modified
        time_modified = os.path.getmtime(RAW_DIR)
        if time_modified == last_time_modified:
            print("No new files found")
            continue
        last_time_modified = time_modified

        files = get_files()
        for file in files:
            matches = re.search(r"VID_(\d{8}_\d{6})_00_(\d{3}).insv", file)
            if not matches:
                continue

            timestamp = matches.group(1)
            file1  = os.path.join(RAW_DIR, "VID_" + timestamp + "_00_" + matches.group(2) + ".insv")
            file2  = os.path.join(RAW_DIR, "VID_" + timestamp + "_10_" + matches.group(2) + ".insv")
            output = os.path.join(OUT_DIR, "VID_" + timestamp + ".mp4")
            file = (file1, file2, output, timestamp)

            if  is_writing(file1) or is_writing(file2):
                print("Files still being written, skipping")
                continue

            # check if file1 and file2 size is the same within 20%
            if abs(os.path.getsize(file1) - os.path.getsize(file2)) > 0.2 * os.path.getsize(file1):
                print("Files are not the same size, skipping")
                continue

            #check if output file exists and size if bigger than file1+file2
            if os.path.exists(output) and os.path.getsize(output) > os.path.getsize(file1) + os.path.getsize(file2):
                print("Already processed, skipping file: " + output)
                continue

            process_file(file)

        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()

