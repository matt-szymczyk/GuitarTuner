import numpy as np
from scipy.fftpack import fft
import pyaudio
import tkinter as tk

class TunerApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.sample_rate = 44100
        self.chunk = 2048
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        self.note_label = tk.Label(self, text="Tuning...")
        self.note_label.pack()

        self.frequency_label = tk.Label(self, text="Frequency:")
        self.frequency_label.pack()

        self.update()

    def update(self):
        data = np.fromstring(self.stream.read(self.chunk), dtype=np.int16)
        fft_data = np.abs(fft(data))[:self.chunk//2]
        peak_frequency = np.argmax(fft_data) * self.sample_rate / self.chunk
        self.frequency_label.config(text="Frequency: {:.2f}Hz".format(peak_frequency))

        note, error = self.note_from_frequency(peak_frequency)
        self.note_label.config(text="Note: {} (error: {:.2f} cents)".format(note, error))
        self.after(50, self.update)

    def note_from_frequency(self, frequency):
        """
        Returns the nearest note and the error (in cents)
        for a given frequency.
        """
        notes = ["E", "F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#"]
        note_frequencies = [82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56]
        error = 1000000
        note = ""
        for i, freq in enumerate(note_frequencies):
            if abs(frequency - freq) < error:
                error = abs(frequency - freq)
                note = notes[i]
        error = error / freq * 1200
        return note, error

if __name__ == "__main__":
    app = TunerApp()
    app.mainloop()
