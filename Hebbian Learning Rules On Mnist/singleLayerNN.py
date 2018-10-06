# Thakur, Nishant
# 1001-544-591
# 2018-10-05
# Assignment-03-01

import sys
import numpy as np
import scipy.misc

if sys.version_info[0] < 3:
    from Tkinter import *
else:
    from tkinter import *


def read_image_and_convert_to_vector(file_name):
    img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
    return img.reshape(-1, 1)  # reshape to column vector and return it


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Hebbian Learning Rules on MNIST")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a button instance
        quitButton = Button(self, text="Exit", command=self.client_exit)

        # placing the button on my window
        quitButton.place(x=0, y=0)

    def client_exit(self):
        exit()


if __name__ == "__main__":
    root = Tk()
    # size of the window
    root.geometry("400x300")
    app = Window(root)
    root.mainloop()
