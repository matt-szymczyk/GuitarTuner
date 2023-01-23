
![Logo](https://i.postimg.cc/PrQmy8Gb/image.png)


# HPS Guitar Tuner

Harmonic Product Spectrum (HPS) jest metodą analizy widmowej sygnału, która polega na mnożeniu widma sygnału z jego własnymi harmonicznymi wersjami.

Proces polega na wykonaniu FFT na sygnale, następnie dzieleniu widma na kilka równych częstotliwościowo przedziałów, a następnie mnożeniu każdego z tych przedziałów przez siebie.

Na przykład, jeśli dzielimy widmo sygnału na trzy równe przedziały, to pierwszy przedział zostanie pomnożony przez drugi, drugi przez trzeci, a trzeci przez pierwszy.

Rezultat końcowy to widmo, które ma wyraźniejsze piksy na częstotliwościach głównych składników sygnału. HPS jest często używany do analizy sygnałów audio, takich jak muzyka, gdzie jest on skuteczny w wykrywaniu tonów podstawowych i harmonicznych.

![App Screenshot](https://i.postimg.cc/FKq46Zd2/image.png)
## Screenshots

![App Screenshot](https://i.postimg.cc/RCjf4LmL/image.png)

![App Screenshot](https://i.postimg.cc/15hgLR0W/image.png)


## Run Locally

Clone the project

```bash
  git clone https://github.com/Matt-Szymczyk/GuitarTuner.git
```

Go to the project directory

```bash
  cd my-project
```

Install libraries

```bash
  pip3 install -r requirements.txt
```

Start the app

```bash
  python3 main.py
```

