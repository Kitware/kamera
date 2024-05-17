import os
import shutil
import glob
import socket
import re
import time
import threading
import datetime
import errno
import psutil
import subprocess
from pathlib import Path

import image_queue
from roskv.impl.redis_envoy import RedisEnvoy


hostname = socket.gethostname()
redis_host = os.environ.get('REDIS_HOST', 'nuvo0')
envoy = RedisEnvoy(redis_host, client_name='image_manager')

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# Ported over from GUI to maintain parity in filepaths
PAT_BRACED = re.compile(r'\{(\w+)\}')
def get_template_keys(tmpl):
    return re.findall(PAT_BRACED, tmpl)

def conformKwargsToFormatter(tmpl, kwargs):
    # type: (str, dict) -> dict
    required_keys = set(get_template_keys(tmpl))
    missing_keys = required_keys.difference(kwargs)
    fmt_dict = {k:v for k,v in kwargs.items() if k in required_keys}
    fmt_dict.update({k:'({})'.format(k) for k in missing_keys})
    return fmt_dict

def get_arch_path(**kwargs):
    arch_dict = envoy.get("/sys/arch")
    tmpl = arch_dict['base_template']
    # Offer manual override
    for k,v in kwargs.items():
        arch_dict[k] = v
    fmt_dict = conformKwargsToFormatter(tmpl, arch_dict)
    return tmpl.format(**fmt_dict)


class ImageManager(object):
    def __init__(self):
        self.lock = threading.Lock()
        self._image_exts = ['jpg', 'tif', 'json']
        self.modalities = ['rgb', 'ir', 'uv', 'meta']
        self.modality2ext = {'rgb':'jpg', 'ir':'tif', 'uv':'jpg', 'meta':'json'}
        self._effort = envoy.kv.get("/sys/arch/effort")
        self._wait_time_sec = int(envoy.kv.get("/sys/effort_metadata_dict/%s/wait_time_sec" % self._effort))
        self._delete_old_images_sec = int(envoy.kv.get("/sys/effort_metadata_dict/%s/delete_old_images_sec" % self._effort))
        self._save_every_x_image = int(envoy.kv.get("/sys/effort_metadata_dict/%s/save_every_x_image" % self._effort))
        self._log_file = '/root/kamera_ws/image_manager.log'
        self._image_dir = ""
        self._secondary_image_dir = ""
        self._detection_dir = ""
        self._secondary_detection_dir = ""

        self.frame_index = 0
        self.dets = set()
        self.processed = set()
        self.modulo_n = set()
        self.previous_imgs_copied = set()
        self.stop_threads = False
        self.previous_detection_dir = ""
        self.thread = None

    @property
    def effort(self):
        try:
            self._effort = envoy.get("/sys/arch/effort")
        except:
            self._effort = "ON"
        return self._effort

    @property
    def wait_time_sec(self):
        try:
            self._wait_time_sec = int(envoy.get(
                "/sys/effort_metadata_dict/%s/wait_time_sec"
                % self.effort))
        except Exception as e:
            self._wait_time_sec = 60
        return self._wait_time_sec

    @property
    def delete_old_images_sec(self):
        try:
            self._delete_old_images_sec = int(envoy.kv.get(
                "/sys/effort_metadata_dict/%s/delete_old_images_sec"
                % self.effort))
        except Exception as e:
            self._delete_old_images_sec = 60
        return self._delete_old_images_sec

    @property
    def save_every_x_image(self):
        try:
            self._save_every_x_image = int(envoy.kv.get(
                "/sys/effort_metadata_dict/%s/save_every_x_image"
                % self.effort))
        except:
            # Default to save every image
            self._save_every_x_image = 1
        return self._save_every_x_image

    @property
    def image_dir(self):
        primary_storage = envoy.kv.get("/sys/local_ssd_mnt")
        syscfg_dir = get_arch_path(base=primary_storage)
        view = envoy.kv.get("/sys/arch/hosts/%s/fov" % hostname) + "_view"
        self._image_dir = os.path.join(syscfg_dir, view)
        return self._image_dir

    @property
    def secondary_image_dir(self):
        secondary_storage = envoy.kv.get("/sys/nas_mnt")
        syscfg_dir = get_arch_path(base=secondary_storage)
        view = envoy.kv.get("/sys/arch/hosts/%s/fov" % hostname) + "_view"
        self._secondary_image_dir = os.path.join(syscfg_dir, view)
        return self._secondary_image_dir

    @property
    def flight_dir(self):
        d = os.path.dirname(os.path.dirname(self.image_dir))
        return d

    @property
    def secondary_flight_dir(self):
        d = os.path.dirname(os.path.dirname(self.secondary_image_dir))
        return d

    @property
    def detection_dir(self):
        d = os.path.dirname(os.path.dirname(self.image_dir))
        self._detection_dir = os.path.join(d, "detections")
        return self._detection_dir

    @property
    def secondary_detection_dir(self):
        d = os.path.dirname(os.path.dirname(self.secondary_image_dir))
        self._secondary_detection_dir = os.path.join(d, "detections")
        return self._secondary_detection_dir

    def get_file_name_parts(self, full_filename):
        name, us, ext = full_filename.split("/")[-1].split(".")
        fname = name + "." + us
        folder = os.path.dirname(full_filename)
        bname = fname.split("_")[0:-1]
        bname = "_".join(bname)
        ftype = fname.split("_")[-1]
        return (folder, fname, bname, ext, ftype)

    def watch_files(self, files):
        print("Opening files.")
        open_files = []
        last_name = ""
        for f in files:
            open_files.append(open(f, "r"))
        while not self.stop_threads:
            if not self.lock.locked():
                self.lock.acquire()
            det_files = glob.glob(self.detection_dir + "/*.csv")
            det_files += glob.glob(self.detection_dir + "/*.txt")
            for f1 in det_files:
                if f1 not in files:
                    open_files.append(open(f1, "r"))
                    files.append(f1)
            empty = 0
            for i, file in enumerate(open_files):
                line = file.readline()
                if not line:
                    empty += 1
                    continue
                if ".csv" in files[i]:
                    if len(line) < 1 or line[0] == "#":
                        continue
                    fullname = line.split(',')[1].strip()
                    folder, fname, bname, ext, ftype = self.get_file_name_parts(
                                                                       fullname)
                    self.dets.add(bname)
                elif ".txt" in files[i]:
                    fullname = line.strip()
                    folder, fname, bname, ext, ftype = self.get_file_name_parts(
                                                                       fullname)
                    if bname not in self.processed and bname not in self.modulo_n:
                        self.frame_index += 1
                    if (self.frame_index % self.save_every_x_image) != 0:
                        # Only add to processed if not part of the nth
                        self.processed.add(bname)
                        try:
                            # Reconsider these images again since they've
                            # now been detected on, possible they'll
                            # get deleted
                            self.previous_imgs_copied.remove(fullname)
                        except KeyError:
                            pass
                    else:
                        self.modulo_n.add(bname)
            # If all files were empty, sleep a bit
            if empty == len(open_files):
                # Only release when files have been read
                if self.lock.locked():
                    self.lock.release()
                time.sleep(0.5)

        print("Closing files")
        for f in open_files:
            f.close()

    def copy_files(self):
        continue_looping = True
        os.umask(0)

        print("Wait time in seconds: %s" % self.wait_time_sec)
        print("Delete old images time in seconds: %s" % self.delete_old_images_sec)
        print("Save every x image: %s" % self.save_every_x_image)
        print("Image dir: %s" % self.image_dir)
        print("Secondary image dir: %s" % self.secondary_image_dir)
        print("Detection dir: %s" % self.detection_dir)
        print("Secondary detection dir: %s" % self.secondary_detection_dir)

        while continue_looping:
            if not os.path.exists("/mnt/flight_data/.flight_data_mounted"):
                raise SystemError("NAS not mounted! Skipping.")
            image_dir = self.image_dir
            detection_dir = self.detection_dir
            secondary_image_dir = self.secondary_image_dir
            secondary_detection_dir = self.secondary_detection_dir
            to_copy = set()
            imgs_to_copy = set()
            det_files = glob.glob(self.detection_dir + "/*.csv")
            det_files += glob.glob(self.detection_dir + "/*.txt")
            # Copy over log file too
            log_files = glob.glob(self.flight_dir+ "/*.txt")
            if detection_dir != self.previous_detection_dir:
                with self.lock:
                    self.stop_threads = True
                    if self.thread is not None:
                        self.thread.join()
                    self.dets.clear()
                    self.processed.clear()
                    self.modulo_n.clear()
                    self.previous_imgs_copied.clear()
                    self.stop_threads = False
                    self.thread = threading.Thread(target=self.watch_files,
                            args=(det_files,))
                    self.thread.daemon = True
                    # Start filling up dets from csvs
                    print("Pulling from detection files...")
                    self.thread.start()
                time.sleep(1)
            with self.lock:
                ims = len(self.processed)
                print("Number of images detected on: %s" % ims)
                print("Number of detections %s." % len(self.dets))
            # Always place detection files in queue since they
            # are constantly being written to
            [ to_copy.add(det) for det in det_files ]
            [ to_copy.add(lf) for lf in log_files ]
            self.previous_detection_dir = detection_dir
            try:
                os.makedirs(secondary_image_dir, 0o777)
            except (OSError, IOError) as exception:
                if exception.errno != errno.EEXIST:
                    raise
            print("Checking dir: %s" % image_dir)
            # get the current image files
            impath = Path(image_dir)
            images = list(impath.glob("*"))
            images = set([str(im) for im in images])
            print("Images globbed.")
            # Only consider images that haven't been copied yet
            with self.lock:
                noncopied_images = list(images.difference(
                                        self.previous_imgs_copied))

            if len(noncopied_images) > 0:
                # We have to copy/delete these still, so mark as processing
                envoy.put("/stat/image_manager/processing", 1)
            else:
                envoy.put("/stat/image_manager/processing", 0)

            # add them to the image queue
            try:
                os.makedirs(secondary_detection_dir, 0o777)
            except (OSError, IOError) as exception:
                if exception.errno != errno.EEXIST:
                    raise

            # remove files from the list that are not older than the wait time
            noncopied_paths = [Path(f) for f in noncopied_images]
            present = time.time()
            process_time = present - self.wait_time_sec
            # only look at files if they are older than n seconds
            files_to_process = sorted([ p for p in noncopied_paths
                                 if p.stat().st_mtime - process_time < 0 ])
            print(f"files left = %s" % len(files_to_process))

            # loop through the files and see if we need to copy/delete
            to_touch = set()
            last_saved_index = 0
            last_name = ""
            print("Images queued.")
            for img in files_to_process:
                mtime = img.stat().st_mtime
                fullname = str(img.resolve())
                # get the name without the extension and ir/uv/rgb component
                folder, fname, bname, ext, ftype = self.get_file_name_parts\
                                                        (fullname)
                #print(bname + "." + ext + " from " + folder + " type " + ftype)

                # see if we have a match with the detections
                with self.lock:
                    if bname in list(self.dets):
                        #print("Added because it has a detection.")
                        imgs_to_copy.add(fullname)
                        continue

                # If we've waited the minimum amount of time and no
                # detection has been generated, copy over as well.
                with self.lock:
                    # If modified time is older than current, copy
                    if bname in self.processed and ext != "json":
                        #print("Adding %s to touch." % bname)
                        # if we made it here then this needs to be deleted
                        # this means this image is not amongst
                        # those being copied over. Leave meta.json alone
                        to_touch.add(fullname)
                    else:
                        #print("Added because it hasn't been detected on in too long,"
                        #      " or is part of modulo N.")
                        imgs_to_copy.add(fullname)

            # For files that are not copied over, we will create a dummy file
            # so we can still generate the gis footprint.
            print("Number of files to touch : %s" % len(to_touch))
            for fname in list(to_touch):
                fnamebase = os.path.basename(fname)
                new_file = os.path.join(secondary_image_dir, fnamebase)
                try:
                    if os.path.exists(new_file):
                        # If it exists on remote, delete first, then touch
                        os.remove(new_file)
                    Path(new_file).touch()
                except Exception as e:
                    print(e)

            # copy files
            for cfile in list(to_copy):
                bname = cfile.split("/")[-1]
                if ".csv" in bname or ".txt" in bname:
                    if "log" in bname:
                        new_file = os.path.join(
                                self.secondary_flight_dir, bname)
                    else:
                        new_file = os.path.join(secondary_detection_dir, bname)
                else:
                    new_file = os.path.join(secondary_image_dir, bname)
                try:
                    # Only copy file if modification times differ
                    if os.path.exists(new_file):
                        if (os.stat(cfile).st_mtime - os.stat(new_file).st_mtime) > 0:
                            shutil.copy2(cfile, new_file)
                        else:
                            print("File exists with the same modification time!")
                    else:
                        shutil.copy2(cfile, new_file)
                except (IOError, OSError) as e:
                    print(f"Error: {e.filename}, {e.strerror}")
                    continue
            print("Number of files to copy: %s" % len(imgs_to_copy))
            csz = 10000
            for chunk in chunker(list(imgs_to_copy), csz):
                rsync_bash = ["rsync", "-a"]
                rsync_bash += chunk
                rsync_bash += [secondary_image_dir]
                print("Running rsync on chunk...")
                try:
                    subprocess.run(rsync_bash)
                except subprocess.CalledProcessError as e:
                    print("Subprocess call failed.")
                    print(e)
            print("Finished rsync!")
            # Keeps a rolling window of what's been copied but not what's
            # been deleted
            with self.lock:
                self.previous_imgs_copied = (self.previous_imgs_copied.union(
                                        imgs_to_copy).union(to_touch)).intersection(images)
            print("Length of prev copied; %s" % len(self.previous_imgs_copied))

            present = time.time()
            old_time = present - self.delete_old_images_sec
            with self.lock:
                sorted_previous_imgs = sorted(self.previous_imgs_copied)
            for fp in sorted_previous_imgs:
                # If it's been copied, and older than time, delete
                img = Path(fp)
                mtime = img.stat().st_mtime
                if mtime < old_time:
                    folder, fname, bname, ext, ftype = self.get_file_name_parts\
                                                        (fp)
                    fpbase = os.path.basename(fp)
                    mntfname = os.path.join(secondary_image_dir, fpbase)
                    # Confirm file was copied before removing, else try to
                    # copy again
                    if os.path.exists(mntfname):
                        try:
                            os.remove(fp)
                        except Exception as e:
                            print(f"Error: {e.filename}, {e.strerror}")
                        try:
                            self.processed.remove(bname)
                        except Exception as e:
                            pass
                        try:
                            self.dets.remove(bname)
                        except Exception as e:
                            pass
                        try:
                            self.modulo_n.remove(fp)
                        except Exception as e:
                            pass
                    else:
                        with self.lock:
                            # Add image back to retry copy
                            self.previous_imgs_copied.remove(fp)
                else:
                    # Since we're sorted, break upon finding a file newer than
                    # parameter
                    break
            time.sleep(1)


if __name__ == "__main__":
    im = ImageManager()
    im.copy_files()
    # save the log from the image queue
    #with open(_log_file, 'w') as jfile:
    #    jfile.write(js)
    # end while
