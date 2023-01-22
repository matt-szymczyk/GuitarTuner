import tkinter as tk
from tkinter import ttk
import math
import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

# General settings that can be changed by the user
SAMPLE_FREQ = 48000 # sample frequency in Hz
WINDOW_SIZE = 48000 # window size of the DFT in samples
WINDOW_STEP = 12000 # step size of window
NUM_HPS = 5 # max number of harmonic product spectrums
POWER_THRESH = 1e-6 # tuning is activated if the signal power exceeds this threshold
CONCERT_PITCH = 440 # defining a1
WHITE_NOISE_THRESH = 0.2 # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE # frequency step width of the interpolated DFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]
def find_closest_note(pitch):
  """
  This function finds the closest note for a given pitch
  Parameters:
    pitch (float): pitch given in hertz
  Returns:
    closest_note (str): e.g. a, g#, ..
    closest_pitch (float): pitch of the closest note in hertz
  """
  i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
  closest_note = ALL_NOTES[i%12] + str(4 + (i + 9) // 12)
  closest_pitch = CONCERT_PITCH*2**(i/12)
  return closest_note, closest_pitch

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Guitar Tuner")
        self.geometry("400x200")
        self.closest_note = tk.StringVar()
        self.max_freq = tk.StringVar()
        self.closest_pitch = tk.StringVar()
        self.arrow_angle = tk.IntVar()
        self.arrow_angle.set(0)

        self.note_label = ttk.Label(self, textvariable=self.closest_note)
        self.note_label.pack()

        self.freq_label = ttk.Label(self, textvariable=self.max_freq)
        self.freq_label.pack()

        self.pitch_label = ttk.Label(self, textvariable=self.closest_pitch)
        self.pitch_label.pack()

        self.canvas = tk.Canvas(self, width=150, height=150)
        self.canvas.pack()

        self.line = self.canvas.create_line(75, 75, 75, 25, width=2)

    def update_labels(self, closest_note, max_freq, closest_pitch):
        self.closest_note.set(closest_note)
        self.max_freq.set(max_freq)
        self.closest_pitch.set(closest_pitch)

    def update_arrow(self, angle):
        self.canvas.delete(self.line)
        x1, y1, x2, y2 = 75, 75, 75 + 50 * math.sin(math.radians(angle)), 75 - 50 * math.cos(math.radians(angle))
        self.line = self.canvas.create_line(x1, y1, x2, y2, width=2)

HANN_WINDOW = np.hanning(WINDOW_SIZE)
def callback(indata, frames, time, status):
    """
  Callback function of the InputStream method.
  That's where the magic happens ;)
  """
    # define static variables
    if not hasattr(callback, "window_samples"):
        callback.window_samples = [0 for _ in range(WINDOW_SIZE)]
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = ["1", "2"]
    if not hasattr(callback, "app"):
        callback.app = App()

    if status:
        print(status)
        return
    if np.any(indata):
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0]))  # append new samples
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]  # remove old samples

        # skip if signal power is too low
        signal_power = (np.linalg.norm(callback.window_samples, ord=2)**2) / len(callback.window_samples)
        if signal_power < POWER_THRESH:
          callback.app.update_labels("...", "...", "...")
          callback.app.update_arrow(0)
          return
        # avoid spectral leakage by multiplying the signal with a hann window
        hann_samples = callback.window_samples * HANN_WINDOW
        magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])

        # supress mains hum, set everything below 62Hz to zero
        magnitude_spec[:31] = 0

        # create harmonic product spectrums
        hps = np.zeros(len(magnitude_spec))
        hps += magnitude_spec
        for i in range(2, NUM_HPS + 1):
            hps += np.interp(i * magnitude_spec.shape[0], np.arange(magnitude_spec.shape[0]), magnitude_spec)

        # set everything under WHITE_NOISE_THRESH*avg_energy_per_freq to zero
        avg_energy_per_freq = (np.sum(hps) / hps.shape[0]) * WHITE_NOISE_THRESH
        hps[hps < avg_energy_per_freq] = 0

        # find maximum frequency
        max_index = np.argmax(hps)
        max_freq = max_index * DELTA_FREQ
        closest_note, closest_pitch = find_closest_note(max_freq)

        callback.app.update_labels(closest_note, max_freq, closest_pitch)
        # you can use the max_freq and closest_pitch values to calculate the angle of the arrow
        angle = calculate_angle(max_freq, closest_pitch)
        callback.app.update_arrow(angle)


def calculate_angle(max_freq, closest_pitch):
    diff = abs(max_freq - closest_pitch)
    angle = 90 - (diff / closest_pitch) * 90
    return angle


if __name__ == "__main__":
    app = App()

    stream = sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ)
    stream.start()
    app.mainloop()
