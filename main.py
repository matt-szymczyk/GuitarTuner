import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
import math
import tkinter as tk
import customtkinter as ctk

# ctk.set_widget_scaling(1.5)  # widget dimensions and text size
# # ctk.set_spacing_scaling(1.5)  # padding and place positions
# ctk.set_window_scaling(1.5)  # window geometry dimensions

SAMPLE_FREQ = 48000  # sample frequency in Hz
WINDOW_SIZE = 48000  # window size of the DFT in samples
WINDOW_STEP = 12000  # step size of window
NUM_HPS = 5  # max number of harmonic product spectrums
POWER_THRESH = 1e-6  # tuning is activated if the signal power exceeds this threshold
CONCERT_PITCH = 440  # defining a1
WHITE_NOISE_THRESH = 0.2  # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ  # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ  # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

HANN_WINDOW = np.hanning(WINDOW_SIZE)


class TunerApp:
    def __init__(self, master):
        self.master = master

        master.title("Guitar Tuner")
        master.geometry("300x500")

        self.tabview = ctk.CTkTabview(master, width=250)
        # self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        self.tuner_tab = self.tabview.add("Tuner")
        self.settings_tab = self.tabview.add("Settings")
        self.tabview.set("Tuner")
        self.tabview.pack(padx=20, pady=20)

        self.scaling_label = ctk.CTkLabel(self.settings_tab, text="Theme:", anchor="w")
        self.scaling_label.grid(row=0, column=0, padx=(0, 20), pady=(0, 20), sticky="nsew")
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.settings_tab,
                                                             values=["System", "Light", "Dark"],
                                                             command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=0, column=0, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.appearance_mode_optionemenu.set("System")

        self.scaling_label = ctk.CTkLabel(self.settings_tab, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=1, column=0, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.scaling_optionemenu = ctk.CTkOptionMenu(self.settings_tab,
                                                     values=["75%", "100%", "125%", "150%", "175%", "200%"],
                                                     command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=1, column=0, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.scaling_optionemenu.set("100%")
        self.scale_factor = 1

        # Closest note
        self.closest_note_label = ctk.CTkLabel(self.tuner_tab, text="Closest note:")
        self.closest_note_label.pack()
        # Closest note value
        self.closest_note_var = tk.StringVar()
        self.closest_note_var.set("...")
        self.closest_note_display = ctk.CTkLabel(self.tuner_tab, textvariable=self.closest_note_var)
        self.closest_note_display.pack()

        # Max frequency label
        self.max_freq_label = ctk.CTkLabel(self.tuner_tab, text="Max frequency:")
        self.max_freq_label.pack()
        # Max frequency value
        self.max_freq_var = tk.StringVar()
        self.max_freq_var.set("...")
        self.max_freq_display = ctk.CTkLabel(self.tuner_tab, textvariable=self.max_freq_var)
        self.max_freq_display.pack()

        # Pitch detection
        self.closest_pitch_label = ctk.CTkLabel(self.tuner_tab, text="Closest pitch:")
        self.closest_pitch_label.pack()
        # Pitch detection value
        self.closest_pitch_var = tk.StringVar()
        self.closest_pitch_var.set("...")
        self.closest_pitch_display = ctk.CTkLabel(self.tuner_tab, textvariable=self.closest_pitch_var)
        self.closest_pitch_display.pack()

        # create canvas
        self.canvas = tk.Canvas(self.tuner_tab, width=150, height=150, bg="black")
        self.canvas.pack()

        # create arrow
        self.line = self.canvas.create_line(75, 75, 75, 25, width=2, fill="red")

        # Start the audio stream button
        self.start_button = ctk.CTkButton(self.tuner_tab, text="Start", command=self.start)
        self.start_button.pack(padx=10, pady=10)

        # Stop the audio stream button
        self.stop_button = ctk.CTkButton(self.tuner_tab, text="Stop", command=self.stop)
        self.stop_button.pack(padx=10, pady=10)

        # Create the audio stream
        self.window_samples = [0 for _ in range(WINDOW_SIZE)]
        self.noteBuffer = ["1", "2"]
        self.stream = sd.InputStream(channels=1, callback=self.callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)
        # scale canvas
        self.canvas.config(width=150 * new_scaling_float, height=150 * new_scaling_float)
        # scale arrow
        self.canvas.coords(self.line, 75 * new_scaling_float, 75 * new_scaling_float, 75 * new_scaling_float, 25 * new_scaling_float)
        self.scale_factor = new_scaling_float
        self.master.geometry(f"300x{int(500 * new_scaling_float)}")
        self.master.geometry(f"{int(300 * new_scaling_float)}x{int(500 * new_scaling_float)}")



    def note_to_pitch(self, note):
        """
        This function finds the pitch of a given note
        Parameters:
            note (str): note given in format "A2" or "G#2"
        Returns:
            pitch (float): pitch of the note in hertz
        """
        # Split the input string into note name and octave
        note_name = note[:-1]
        octave = int(note[-1])

        # Get the index of the note in ALL_NOTES
        note_index = ALL_NOTES.index(note_name)

        # Calculate the number of semitones above A1
        semitones_above_a1 = 12 * (octave - 1) + note_index

        # Calculate the pitch
        pitch = CONCERT_PITCH * 2 ** (semitones_above_a1 / 12)

        return pitch

    def update_arrow(self, angle):
        self.canvas.delete(self.line)
        x1, y1, x2, y2 = 75, 75, 75 + 50 * math.sin(math.radians(angle)), 75 - 50 * math.cos(math.radians(angle))
        sf = self.scale_factor
        self.line = self.canvas.create_line(x1 * sf, y1 * sf, x2 * sf, y2 * sf, width=2, fill="red")

    def calculate_angle(self, max_freq, closest_pitch):
        diff = max_freq - closest_pitch
        angle = (diff / closest_pitch) * 1800
        print(angle)
        angle = min(180, max(0, angle))
        return angle

    def find_closest_note(self, pitch):
        """
        This function finds the closest note for a given pitch
        Parameters:
          pitch (float): pitch given in hertz
        Returns:
          closest_note (str): e.g. a, g#, ..
          closest_pitch (float): pitch of the closest note in hertz
        """
        i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
        closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
        closest_pitch = CONCERT_PITCH * 2 ** (i / 12)

        return closest_note, closest_pitch

    def callback(self, indata, frames, time, status):
        """
        Callback function of the InputStream method.
        That's where the magic happens ;)
        """
        # define static variables
        if not hasattr(self.callback, "window_samples"):
            self.window_samples = [0 for _ in range(WINDOW_SIZE)]
        if not hasattr(self.callback, "noteBuffer"):
            self.noteBuffer = ["1", "2"]

        if status:
            print(status)
            return
        if any(indata):
            self.window_samples = np.concatenate((self.window_samples, indata[:, 0]))  # append new samples
            self.window_samples = self.window_samples[len(indata[:, 0]):]  # remove old samples

            # skip if signal power is too low
            signal_power = (np.linalg.norm(self.window_samples, ord=2, axis=0) ** 2) / len(
                self.window_samples)
            if signal_power < POWER_THRESH:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Closest note: ...")
                return

            # avoid spectral leakage by multiplying the signal with a hann window
            hann_samples = self.window_samples * HANN_WINDOW
            magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])

            # supress mains hum, set everything below 62Hz to zero
            for i in range(int(62 / DELTA_FREQ)):
                magnitude_spec[i] = 0

            # calculate average energy per frequency for the octave bands
            # and suppress everything below it
            for j in range(len(OCTAVE_BANDS) - 1):
                ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
                ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
                ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
                avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2, axis=0) ** 2) / (
                        ind_end - ind_start)
                avg_energy_per_freq = avg_energy_per_freq ** 0.5
                for i in range(ind_start, ind_end):
                    magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[
                                                                 i] > WHITE_NOISE_THRESH * avg_energy_per_freq else 0

            # interpolate spectrum
            mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS),
                                      np.arange(0, len(magnitude_spec)),
                                      magnitude_spec)
            mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2, axis=0)  # normalize it

            hps_spec = copy.deepcopy(mag_spec_ipol)

            # calculate the HPS
            for i in range(NUM_HPS):
                tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))],
                                           mag_spec_ipol[::(i + 1)])
                if not any(tmp_hps_spec):
                    break
                hps_spec = tmp_hps_spec

            max_ind = np.argmax(hps_spec)
            max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

            closest_note, closest_pitch = self.find_closest_note(max_freq)
            max_freq = round(max_freq, 1)
            closest_pitch = round(closest_pitch, 1)

            self.max_freq_var.set(max_freq)
            self.closest_note_var.set(closest_note)
            self.closest_pitch_var.set(closest_pitch)

            self.update_arrow(self.calculate_angle(max_freq, closest_pitch))

            self.noteBuffer.insert(0, closest_note)  # note that this is a ringbuffer
            self.noteBuffer.pop()

            os.system('cls' if os.name == 'nt' else 'clear')
            if self.noteBuffer.count(self.noteBuffer[0]) == len(self.noteBuffer):
                print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
            else:
                print(f"Closest note: ...")

        else:
            print('no input')

    def start(self):
        """
        Start the audio stream
        """

        self.stream.start()

    def stop(self):
        """
        Stop the audio stream
        """
        self.stream.stop()
        # self.stream.close()
        # self.stream = None


if __name__ == "__main__":
    root = ctk.CTk()
    app = TunerApp(root)
    app.start()

    root.mainloop()
