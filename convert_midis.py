import os
import random
from pathlib import Path

target_dir = input("Enter Directory to Convert (No Quotes Needed): ")
composer_name = input("Enter name of composer subfolders to add to / make: ")

for dname in ['validation','test','train']:
    Path(f"./midis/{dname}/{composer_name}").mkdir(parents=True,exist_ok=True)

for (dirpath, dirnames, filenames) in os.walk(target_dir):
    for filename in [f for f in filenames if f.endswith('.mid')]:
        print(f"Found file {filename}")
        # print(f'Command: midicsv "{dirpath}\{filename}" ')
        out_file_name = filename[:-4]
        target_filename = rf"{dirpath}\{filename}"

        x = random.randint(1,20)
        if x >= 1 and x <= 3:
            outfile_path = rf".\midis\validation\{composer_name}\{out_file_name}.csv"
        elif x >= 4 and x <= 6:
            outfile_path = rf".\midis\test\{composer_name}\{out_file_name}.csv"
        else:
            outfile_path = rf".\midis\train\{composer_name}\{out_file_name}.csv"

        os.system(f'midicsv "{target_filename}" "{outfile_path}"')
