import os


def read_all_pth():
    files = [fn for fn in os.listdir(".") if fn.endswith(".pth")]
    print("Read %d files." % len(files))
    return files

def new_name(checkpoint_name):
    chk_num = int((checkpoint_name[:-4]).split("_")[1])
    print("Checkpoint name: '%s', parsed %d" % (checkpoint_name, chk_num))
    if chk_num > 999:
        print("Stopping, chk_num > 999")
    ret = "checkpoint_%03d.pth" % chk_num
    print("new name is: %s" % ret)
    return ret


if __name__ == "__main__":
    files = read_all_pth()
    for checkpoint_name in files:
        renamed = new_name(checkpoint_name)
        #quit()
        os.rename(checkpoint_name, renamed)



