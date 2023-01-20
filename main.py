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
        Returns the nearest note, the error (in cents) and the string
        for a given frequency.
        """
        note_frequencies = {'E2': 82.41, 'A2': 110.00, 'D3': 146.83, 'G3': 196.00, 'B3': 246.94, 'E4': 329.63}
        error = 1000000
        note = ""
        for i, freq in note_frequencies.items():
            if abs(frequency - freq) < error:
                error = abs(frequency - freq)
                note = i
        error = error / freq * 1200
        return note, error


if __name__ == "__main__":
    app = TunerApp()
    app.mainloop()
