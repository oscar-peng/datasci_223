# Demo 4: Dash Speech-to-Text with Waveform Visualization

> **Note:** This notebook-style guide demonstrates how to build a Dash app that transcribes speech from audio files and visualizes the waveform. It uses the Vosk speech recognition library and Plotly for visualization. You can convert this file to a Jupyter notebook with Jupytext.

---

## 1. Install Requirements

Add these to your `requirements.txt` or install directly:

```
pip install dash jupyter-dash plotly soundfile numpy vosk
```

You also need to download a small Vosk model (see next cell).

---

## 2. Download a Small Vosk Model (Automated)

You can run this cell to automatically download and extract the small English Vosk model. (Requires the `requests` package.)

```python
import os
import zipfile
import requests

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_DIR = "vosk-model-small-en-us-0.15"
MODEL_ZIP = MODEL_DIR + ".zip"

def download_and_extract_vosk_model():
    if not os.path.exists(MODEL_DIR):
        print("Downloading Vosk model...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(MODEL_ZIP)
        print("Model downloaded and extracted.")
    else:
        print("Vosk model already present.")

download_and_extract_vosk_model()
```

---

## 3. Imports & Setup

```python
import os
import numpy as np
import soundfile as sf
from vosk import Model, KaldiRecognizer
import json
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, no_update
import base64
import io
```

---

## 4. Helper: Load and Process Audio

```python
def read_audio(file_contents):
    # Decode base64 and read audio file
    content_type, content_string = file_contents.split(',')
    audio_bytes = base64.b64decode(content_string)
    with io.BytesIO(audio_bytes) as f:
        data, samplerate = sf.read(f)
    # If stereo, take one channel
    if data.ndim > 1:
        data = data[:, 0]
    return data, samplerate
```

---

## 4A. Helper: Encode WAV File as Base64 for Dash Audio Player

```python
def wavfile_to_base64(path):
    """Read a WAV file and return a base64-encoded string suitable for Dash audio src."""
    import base64
    with open(path, "rb") as f:
        data = f.read()
    return "data:audio/wav;base64," + base64.b64encode(data).decode()
```

---

## 5. Helper: Transcribe Audio with Vosk

```python
def transcribe_audio(data, samplerate, model_path="vosk-model-small-en-us-0.15"):
    if not os.path.exists(model_path):
        return "Model not found. Please download and unzip the Vosk model."
    model = Model(model_path)
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)
    # Vosk expects 16-bit PCM mono
    import soundfile as sf
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, data, samplerate, subtype='PCM_16')
        with sf.SoundFile(tmp.name) as f:
            while True:
                # FIX: Use f.read(...).tobytes() for Vosk compatibility
                buf = f.read(4000, dtype='int16')
                if len(buf) == 0:
                    break
                if rec.AcceptWaveform(buf.tobytes()):
                    pass
            result = rec.FinalResult()
    text = json.loads(result).get("text", "")
    return text
```

---

## 6. Helper: Plot Waveform

```python
def plot_waveform(data, samplerate):
    duration = len(data) / samplerate
    time = np.linspace(0, duration, len(data))
    fig = go.Figure(go.Scatter(x=time, y=data, mode='lines', name='Waveform'))
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=250,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig
```

---

## 7. Dash App Layout

```python
app = Dash(__name__)
app.layout = html.Div([
    html.H2("Speech-to-Text Demo (Vosk + Dash)"),
    dcc.Upload(
        id='upload-audio',
        children=html.Button('Upload WAV File'),
        accept='.wav',
        multiple=False
    ),
    html.Button('Use Default Sample', id='use-default', n_clicks=0, style={"margin-left": "1em"}),
    html.Div(id='audio-player'),
    dcc.Graph(id='waveform'),
    html.H4("Recognized Text:"),
    html.Pre(id='transcript', style={"background": "#f4f4f4", "padding": "1em"}),
    html.Div("Bonus: Highlighting words on the waveform is possible if word timing is available from Vosk. This is not implemented here, but see Vosk's word-level output for inspiration!", style={"color": "#888", "font-size": "0.9em"})
])
```

---

## 8. Dash Callbacks (with Word Highlighting and Working Audio Player)

```python
from dash import ctx

def get_word_highlight_figure(data, samplerate, words):
    duration = len(data) / samplerate
    time = np.linspace(0, duration, len(data))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name='Waveform'))

    # Add word highlights as semi-transparent rectangles
    for w in words:
        fig.add_vrect(
            x0=w['start'], x1=w['end'],
            fillcolor='orange', opacity=0.3, line_width=0,
        )

    # Place vertical word labels at 80% of the y-axis range for better visibility
    y_max = np.max(data)
    y_min = np.min(data)
    y_label = y_min + 0.8 * (y_max - y_min)
    word_times = [(w['start'] + w['end']) / 2 for w in words]
    word_labels = [w['word'] for w in words]
    fig.add_trace(go.Scatter(
        x=word_times,
        y=[y_label] * len(word_times),
        text=word_labels,
        mode="text",
        textangle=90,
        textfont=dict(color="black", size=12),
        showlegend=False,
        hoverinfo="none"
    ))

    fig.update_layout(
        title="Audio Waveform with Vertical Word Highlights",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

@app.callback(
    [Output('audio-player', 'children'), Output('waveform', 'figure'), Output('transcript', 'children')],
    [Input('upload-audio', 'contents'), Input('use-default', 'n_clicks')],
    prevent_initial_call=False
)
def update_output(contents, n_default):
    trigger = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None
    if (contents is None and (not n_default or n_default == 0)):
        # Nothing to show yet
        return no_update, go.Figure(), ""
    if trigger == 'use-default' or (contents is None and n_default):
        # Use default sample
        wav_path = "data/harvard.wav"
        data, samplerate = sf.read(wav_path)
        audio_src = wavfile_to_base64(wav_path)
    else:
        data, samplerate = read_audio(contents)
        audio_src = contents
    # Transcribe and get word timings
    model_path = "vosk-model-small-en-us-0.15"
    model = Model(model_path)
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, data, samplerate, subtype='PCM_16')
        with sf.SoundFile(tmp.name) as f:
            while True:
                buf = f.read(4000, dtype='int16')
                if len(buf) == 0:
                    break
                rec.AcceptWaveform(buf.tobytes())
            result = rec.FinalResult()
    result_json = json.loads(result)
    words = result_json.get('result', [])
    transcript = result_json.get('text', '')
    # Plot waveform with word highlights
    fig = get_word_highlight_figure(data, samplerate, words)
    audio_player = html.Audio(src=audio_src, controls=True, style={"width": "100%"})
    return audio_player, fig, transcript
```

---

**Now the Dash app will:**
- Show a working audio player for both uploaded and default audio
- Display the waveform
- Highlight recognized words on the waveform
- Show the transcript

Students can upload their own WAV file or use the default sample and see everything in action!

---

## 9. Run the App Inline

```python
app.run(mode='inline', debug=True)
```

---

## 10. Notes & Tips

- Only short WAV files are recommended for demo purposes (a few seconds).
- If you want to highlight words on the waveform, use the `words` field from Vosk's JSON output to get word start/end times and overlay them on the Plotly figure.
- For more languages, download the appropriate Vosk model.
- This demo is fully local and does not require an internet connection after the model is downloaded.

---

## 3A. Demo: Visualize and Transcribe a Sample Audio File (`harvard.wav`)

We'll use the classic `harvard.wav` speech sample (public domain) as a default demo. This cell loads the file, visualizes the waveform, transcribes it, and overlays word highlights for the first 15 seconds.

```python
import soundfile as sf
import numpy as np
import plotly.graph_objs as go
import json
from vosk import Model, KaldiRecognizer

# Load the sample audio
wav_path = "data/harvard.wav"
data, samplerate = sf.read(wav_path)

# Limit to first 15 seconds
max_seconds = 15
if len(data) > samplerate * max_seconds:
    data = data[:samplerate * max_seconds]

# Transcribe with Vosk (word-level)
model_path = "vosk-model-small-en-us-0.15"
model = Model(model_path)
rec = KaldiRecognizer(model, samplerate)
rec.SetWords(True)
import tempfile
with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
    sf.write(tmp.name, data, samplerate, subtype='PCM_16')
    with sf.SoundFile(tmp.name) as f:
        while True:
            # FIX: Use f.read(...).tobytes() for Vosk compatibility
            buf = f.read(4000, dtype='int16')
            if len(buf) == 0:
                break
            rec.AcceptWaveform(buf.tobytes())
        result = rec.FinalResult()
result_json = json.loads(result)
words = result_json.get('result', [])  # List of dicts with 'start', 'end', 'word'
transcript = result_json.get('text', '')

# Plot waveform with word highlights

duration = len(data) / samplerate
time = np.linspace(0, duration, len(data))
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name='Waveform'))

# Add word highlights as semi-transparent rectangles
for w in words:
    fig.add_vrect(
        x0=w['start'], x1=w['end'],
        fillcolor='orange', opacity=0.3, line_width=0,
        annotation_text=w['word'], annotation_position="top left"
    )

fig.update_layout(
    title="Harvard.wav Waveform with Word Highlights",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    height=300,
    margin=dict(l=10, r=10, t=30, b=10)
)
fig.show()

print("Transcript:")
print(transcript)
```

--- 