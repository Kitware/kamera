import glob
import shutil
import time

snap_shot_dir = '/mnt/data/colmap/snapshots/*'
num_to_keep = 10

while True:
    dirs = glob.glob(snap_shot_dir)
    dirs = sorted(dirs)
    print('Found %i subdirectories' % len(dirs))
    for d in dirs[:-num_to_keep]:
        print('Deleting \'%s\'' % d)
        shutil.rmtree(d)

    time.sleep(10)