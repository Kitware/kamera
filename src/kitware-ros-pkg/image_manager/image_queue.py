# a basic state machine to handle the queue of images
import sys
import datetime
import json

class States():
    new_image = "new_image"
    processing = "processing"
    deleted = "deleted"
    copied = "copied"


class Image_State():
    name = ""
    state = None
    init_time = None
    changed_time = None
    detection = None
    time_format = '%Y%m%d_%H%M%S_%f'
    def __init__(self, name):
        self.name = name
        self.state = States.new_image # default state
        self.init_time = self.getTime()
        self.changed_time = None
        self.detection = None

    def getTime(self):
        dt = datetime.datetime.now()
        return dt.strftime(self.time_format)

    # this checks that we have a state that is valid
    def get_props(self, clss):
        return [i for i in clss.__dict__.keys() if i[:1] != '_']

    def change_state(self, newState):
        try:
            # make sure the state is valid
            validProps = self.get_props(States)
            if newState not in validProps:
                print('Error in Image_State: state must be a valid member of States class.')
                return
            # make sure it isn't already there
            if newState == self.state:
                print('Error in Image_State: requested change in state, but it is already at: ' + str(self.state))
                return

            # finally, change the state
            self.changed_time = self.getTime()
            self.state = newState
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print('error in changeState: ' + str(e) + ', line ' + str(exc_tb.tb_lineno) )

    def set_detection(self, true_or_false):
        self.detection = true_or_false

    def get_info(self):
        info = self.name + ","
        info += " init: " + str(self.init_time) + ","
        info += " state: " + str(self.state) + ","
        info += " changed: " + str(self.changed_time)
        info += " detection: " + str(self.detection)
        return info

    def get_dict(self):
        js = {}
        js["name"] = self.name
        js["init_time"] = self.init_time
        js["state"] = self.state
        js["changed_time"] = self.changed_time
        js["detection"] = self.detection
        return js

    def get_json(self):
        js = self.get_dict()
        return json.dumps(js, sort_keys=True, indent=4)


class Image_Queue():
    images = []
    def __init__(self):
        self.images = list()

    def get_names(self):
        return [x.name for x in self.images]

    def get_new_files(self, files):
        # check to see if we already have these files
        prev_files = self.get_names()
        for img in files:
            if img in prev_files:
                continue  # skip files we already have
            new_img = Image_State(img)
            self.images.append(new_img)

    def get_list(self):
        js = []
        for img in self.images:
            js.append(img.get_dict())
        return js

    def get_json(self):
        js = self.get_list()
        return json.dumps(js, sort_keys=True, indent=4)

    def get_image_by_name(self, img_name):
        im = None
        for img in self.images:
            if img_name == img.name:
                im = img
                break

        if im is None:
            print(f"could not find file {img_name}")
        return im

    def change_state(self, img_name, new_state):
        im = self.get_image_by_name(img_name)
        if im is None:
            return
        im.change_state(new_state)

    def set_detection(self, img_name, true_or_false):
        im = self.get_image_by_name(img_name)
        if im is None:
            return
        im.set_detection(true_or_false)


if __name__ == "__main__":

    # ------------ tests------------

    # basic state tests
    img = Image_State("file1.jpg")
    print(f"{img.name} state = {img.state}, created = {img.init_time}")
    print(img.get_info())
    img.change_state(States.deleted)
    print(f"{img.name} state = {img.state}, changed = {img.changed_time}")
    print(img.get_info())

    print(img.get_json())

    # basic image queue tests
    iq = Image_Queue()
    files = ["i1.jpg", "i2.jpg", "i3.jpg"]
    iq.get_new_files(files)
    print(iq.get_names())
    files = ["i1.jpg", "i4.jpg", "i5.jpg"]  # 2 new files - one existing
    iq.get_new_files(files)
    print(iq.get_names())  # we should only see 5 here

    iq.change_state("i2.jpg", States.copied)
    iq.set_detection("i1.jpg", False)
    iq.set_detection("i2.jpg", True)
    print(iq.get_json())
