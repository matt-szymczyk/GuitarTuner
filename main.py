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

        self.label = tk.Label(self, text="Tuning...")
        self.label.pack()

        self.update()

    def update(self):
        data = np.fromstring(self.stream.read(self.chunk), dtype=np.int16)
        fft_data = np.abs(fft(data))[:self.chunk//2]
        peak_frequency = np.argmax(fft_data) * self.sample_rate / self.chunk
        self.label.config(text="Frequency: {:.2f}Hz".format(peak_frequency))
        self.after(50, self.update)

if __name__ == "__main__":
    app = TunerApp()
    app.mainloop()
