"""
This module runs a GUI to play trajectory videos.
It takes trajectory numbers and outputs a window with two videos.

Author: Ruya Karagulle
Date: June 2023
"""

from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkvideo as tv


class GUI:
    def __init__(self):
        pass

    def trajectory1(self):
        """Command executed when trajectory 1 is selected."""
        self.answer = 0
        self.window.destroy()

    def trajectory2(self):
        """Command executed when trajectory 1 is selected."""
        self.answer = 1
        self.window.destroy()

    def center_window(self, window, width=700, height=400):
        """Center the Tkinter window on the screen."""
        # get screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # calculate position x and y coordinates
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        window.geometry("%dx%d+%d+%d" % (width, height, x, y))

    def playVideo_a(self):
        """Play Trajectory A video."""
        self.player_a.play()

    def playVideo_b(self):
        """Play Trajectory B video."""
        self.player_b.play()

    def play_question_videos(self, question_file, selected_q):
        """
        Display a window with two trajectory videos
        based on the selected trajectory numbers.
        """
        self.window = Tk()
        self.window.geometry("1500x600")
        self.center_window(self.window, 1500, 600)
        self.window.title("Driving Behavior Preferences")

        filename_a = (
            question_file
            + f"/trajectory_{selected_q[0]}/trajectory_{selected_q[0]}"
            + ".avi"
        )
        filename_b = (
            question_file
            + f"/trajectory_{selected_q[1]}/trajectory_{selected_q[1]}"
            + ".avi"
        )
        size_adjust = 0.75
        video_size = (int(size_adjust * 1300), int(size_adjust * 724))

        button_a = Button(text="Replay Trajectory A", command=self.playVideo_a)
        button_b = Button(text="Replay Trajectory B", command=self.playVideo_b)

        # button_a.pack(side=LEFT)
        button_a.grid(column=0, row=0, padx=10)
        button_b.grid(column=1, row=0, padx=10)
        # button_b.pack(side=LEFT)

        frame_a = Frame(master=self.window, width=700, height=350)
        frame_b = Frame(master=self.window, width=700, height=350)

        label_a = Label(frame_a, text="Trajectory A")
        label_b = Label(frame_b, text="Trajectory B")

        label_a.pack()
        label_b.pack()

        video_label = Label(frame_a)
        video_label.pack()
        self.player_a = tv.tkvideo(filename_a, label=video_label, size=video_size)
        self.player_a.play()

        video_label = Label(frame_b)
        video_label.pack()
        self.player_b = tv.tkvideo(filename_b, label=video_label, size=video_size)
        self.player_b.play()

        frame_a.pack_propagate(0)
        frame_b.pack_propagate(0)
        # frame_a.pack(side=LEFT, padx=0, pady=0)
        # frame_b.pack(side=RIGHT, padx=0, pady=0)

        frame_a.grid(column=0, row=1, padx=10)
        frame_b.grid(column=1, row=1, padx=10)

        self.window.mainloop()

    def plotGUI(self, fig: Figure):
        """
        Plots the matplotlib Figure to TkInter canvas.0

        Args:
            fig (Figure): matplotlib figure to be plotted
        """
        # Tkinter window and settings
        self.window = Tk()
        self.window.title("System Behaviors")
        self.window.geometry("500x600")

        # canvas for matplotlib plot
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()

        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()

        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.window)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()

        # selection buttons
        A0 = Button(self.window, text="Trajectory 1", fg="blue", command=self.trajectory1)
        A0.pack(side=LEFT, expand=False, fill=None, anchor="center")
        A1 = Button(
            self.window, text="Trajectory 2", fg="orange", command=self.trajectory2
        )
        A1.pack(side=RIGHT, expand=False, fill=None, anchor="center")

        # run the gui
        self.window.mainloop()
