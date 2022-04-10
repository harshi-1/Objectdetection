import sys
import os
from tkinter import *
from tkinter import filedialog

window = Tk()

window.title('Object Detection')

window.geometry('600x500')


def loadfilename():

    filetype = (('Image Files', '*.jpeg;*.jpg;*.png'), ('All files', '*.*'))

    filename = filedialog.askopenfilename(filetypes=filetype)

    script1(filename)


def script1(filename):

    os.system('python detect_image.py ' + filename)


def script2():

    os.system('python detect_video.py')


btn = Button(window, text="Detect From Image", bg="black", fg="white",
             command=loadfilename)

btn.grid(column=0, row=0, padx=100, pady=200)

btn = Button(window, text="Detect From Video",
             bg="black", fg="white", command=script2)

btn.grid(column=4, row=0, padx=100, pady=200)

window.mainloop()
