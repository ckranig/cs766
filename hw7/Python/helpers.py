import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from moviepy.editor import ImageSequenceClip

def generateVideo2(data_params):
    n_frames = len(data_params.frame_ids)
    video_path = os.path.join(data_params.out_dir, 'video.mp4')

    images = [data_params.gen_out_fname(data_params.frame_ids[i]) for i in range(n_frames)]

    clip = ImageSequenceClip(images, fps=30)
    clip.write_videofile(video_path)


def chooseTarget(data_params):
    """
    chooseTarget displays an image and asks the user to drag a rectangle
    around a tracking target

    arguments:
    data_params: a dictionary contains data parameters
    rect: [xmin ymin width height]
    """

    # Reading the first frame from the focal stack
    img_fn = data_params.gen_data_fname(0)

    # Display the image
    # img = Image.open(img_fn)
    # plt.imshow(img)
    # plt.show()

    print('===========')
    print('Drag a rectangle around the tracking target. Close the window once your done!')
    annotator = ImageBoundingBoxAnnotator(img_fn)
    annotator.run()
    rect = annotator.get_rect_coords()

    # To make things easier, let's make the height and width all odd
    if rect[2] % 2 == 0:
        rect[2] += 1
    if rect[3] % 2 == 0:
        rect[3] += 1

    print(f'[xmin ymin width height]  = {rect}')
    print('===========')

    return rect


import tkinter as tk
from PIL import Image, ImageTk


class ImageBoundingBoxAnnotator:
    def __init__(self, img_path):
        self.root = tk.Tk()
        img = Image.open(img_path)
        self.tk_img = ImageTk.PhotoImage(img)  # keep a reference to the image

        self.canvas = tk.Canvas(self.root, width=img.width, height=img.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)

        self.canvas.bind("<Button-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.update_rect)
        self.rect = None
        self.rect_coords = None

    def start_rect(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red')
        self.rect_coords = [event.x, event.y, event.x, event.y]

    def update_rect(self, event):
        if self.rect:
            self.canvas.coords(self.rect, *self.rect_coords)
            self.rect_coords = [self.rect_coords[0], self.rect_coords[1], event.x, event.y]

    def get_rect_coords(self):
        x1, y1, x2, y2 = self.rect_coords
        x_tl, y_tl = min(x1, x2), min(y1, y2)
        x_br, y_br = max(x1, x2), max(y1, y2)
        return [x_tl, y_tl, x_br-x_tl, y_br-y_tl]

    def run(self):
        self.root.mainloop()


def generateVideo(data_params):
    n_frames = len(data_params.frame_ids)
    video_path = os.path.join(data_params.out_dir, 'video.mp4')
    #os.makedirs(data_params.out_dir, exist_ok=True)

    # Loop over all frames
    writer = imageio.get_writer(video_path)
    for i in range(n_frames):
        img_path = data_params.gen_out_fname(data_params.frame_ids[i])
        frame = imageio.imread(img_path)
        writer.append_data(frame)
    writer.close()