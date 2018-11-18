import os
from os.path import join
from util.data_processing import GifReader
from util import paths

if __name__ == "__main__":

    reader = GifReader()
    breakout_dir = join(paths.sequences_dir, "breakout")
    gif_dir = join(breakout_dir, "gifs")
    all_gifs = [join(gif_dir, name) for name in os.listdir(gif_dir)
                                          if name.endswith(".gif")]

    for path in all_gifs:
        outdir = join(breakout_dir, "tmp-output", os.path.basename(path)[:-4] + "_out") 
        try:
            os.mkdir(outdir)
            print("Successfully created %s." % outdir)
        except OSError:
            print("NO! Already exists, %s." % outdir)
            continue
        reader.save_each_frame(path, outdir)

    # Rename all the gifs
    # for path in all_gifs:
    #     print("Doing path:", path)
    #     start, end = map(int, os.path.basename(path)[:-4].split("-"))
    #     newname = "%03d-%03d.gif" % (start, end)
    #     print("New name is:", newname)
    #     #quit()
    #     os.rename(path, join(breakout_dir, newname))