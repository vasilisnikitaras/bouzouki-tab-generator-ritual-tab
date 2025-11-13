# 📦 Εισαγωγή απαραίτητων βιβλιοθηκών
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yt_dlp
import librosa
import soundfile as sf
import os
import re
from fpdf import FPDF
from mido import Message, MidiFile, MidiTrack
from datetime import datetime
import librosa.display

# 🎼 Ορισμός βάσεων χορδών για το τετράχορδο μπουζούκι
string_bases = {'Ντο': 48, 'Φα': 53, 'Λα': 57, 'Ρε': 62}
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# 🔁 Μετατροπή συχνότητας σε MIDI αριθμό
def freq_to_midi(freq):
    return int(round(69 + 12 * np.log2(freq / 440.0)))

# 🔁 Μετατροπή νότας σε MIDI αριθμό (με έλεγχο εγκυρότητας)
def note_to_midi(note):
    match = re.match(r'^([A-G]#?|[A-G]b?)(-?\d+)$', note.strip())
    if not match:
        raise ValueError(f"Μη έγκυρη νότα: {note}")
    name, octave = match.groups()
    octave = int(octave)
    return note_names.index(name) + 12 * (octave + 1)

# 🔁 Μετατροπή MIDI αριθμού σε νότα (π.χ. A4)
def midi_to_note(midi):
    name = note_names[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"

# 🔁 Μετατροπή MIDI σε συχνότητα (Hz)
def midi_to_freq(midi):
    return round(440 * 2 ** ((midi - 69) / 12), 2)

# 🎯 Εύρεση θέσεων (χορδή και τάστο) για συγκεκριμένο MIDI
def find_positions(midi):
    return [(s, midi - b) for s, b in string_bases.items() if 0 <= midi - b <= 12]

# 🎨 Σχεδίαση θέσεων στο μανίκι του μπουζουκιού
def plot_positions(midi):
    positions = find_positions(midi)
    fig, ax = plt.subplots(figsize=(10, 4))
    strings = list(string_bases.keys())
    ax.set_yticks(range(len(strings)))
    ax.set_yticklabels(strings)
    ax.set_xticks(range(13))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, len(strings) - 0.5)
    ax.grid(True)
    for s, f in positions:
        y = strings.index(s)
        ax.plot(f, y, 'ro', markersize=12)
        ax.text(f, y + 0.2, midi_to_note(midi), ha='center')
    st.pyplot(fig)

# 🧠 Δημιουργία ταμπλατούρας από λίστα νοτών και διάρκειας
def tab_from_notes(note_list):
    tab = []
    for note, dur in note_list:
        midi = note_to_midi(note)
        pos = find_positions(midi)
        if pos:
            s, f = pos[0]
            tab.append({'Νότα': note, 'Χορδή': s, 'Τάστο': f, 'Διάρκεια': dur})
        else:
            tab.append({'Νότα': note, 'Χορδή': '—', 'Τάστο': '—', 'Διάρκεια': dur})
    return tab

# 🖼️ Δημιουργία εικόνας με τις θέσεις στο μανίκι
def generate_fretboard_image(tab):
    fig, ax = plt.subplots(figsize=(10, 2))
    strings = list(string_bases.keys())
    ax.set_yticks(range(len(strings)))
    ax.set_yticklabels(strings)
    ax.set_xticks(range(13))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, len(strings) - 0.5)
    ax.grid(True)
    for t in tab:
        if t['Τάστο'] != '—':
            y = strings.index(t['Χορδή'])
            ax.plot(t['Τάστο'], y, 'ro', markersize=10)
            ax.text(t['Τάστο'], y + 0.2, t['Νότα'], ha='center', fontsize=8)
    image_path = "fretboard.png"
    fig.savefig(image_path, bbox_inches='tight')
    plt.close(fig)
    return image_path

# 📄 Δημιουργία PDF με ταμπλατούρα και εξώφυλλο
def generate_pdf(tab):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="🎼 Τελετουργική Ταμπλατούρα για Τετράχορδο Μπουζούκι", ln=True, align='C')
    pdf.ln(10)
    for t in tab:
        line = f"{t['Νότα']} → Χορδή: {t['Χορδή']}, Τάστο: {t['Τάστο']}, Διάρκεια: {t['Διάρκεια']}"
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, txt="📄 Εξώφυλλο", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Τίτλος: Τελετουργική Ταμπλατούρα", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Ημερομηνία: {datetime.now().strftime('%d/%m/%Y')}", ln=True, align='L')
    pdf.cell(200, 10, txt="Υπογραφή: Βασίλης", ln=True, align='L')
    image_path = generate_fretboard_image(tab)
    pdf.image(image_path, x=10, y=pdf.get_y(), w=180)
    pdf.output("tab.pdf")
    return "tab.pdf"

# 🎹 Εξαγωγή MIDI αρχείου από την ταμπλατούρα
def export_midi(tab, filename="output.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    time_unit = 480
    for t in tab:
        try:
            midi = note_to_midi(t['Νότα'])
            duration = int(t['Διάρκεια'] * time_unit)
            track.append(Message('note_on', note=midi, velocity=64, time=0))
            track.append(Message('note_off', note=midi, velocity=64, time=duration))
        except:
            continue
    mid.save(filename)
    return filename

# 📈 Φασματική ανάλυση αρχείου ήχου
def plot_spectrum(file_path):
    y, sr = librosa.load(file_path)
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    ax.set_title("📈 Φασματική Ανάλυση")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

# 📥 Λήψη ήχου από YouTube ως WAV
def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'ffmpeg_location': r'C:\Users\Admin\Downloads\ffmpeg\bin',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'audio.wav'

# 🎵 Εξαγωγή νοτών από αρχείο ήχου
def extract_notes_from_audio(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    notes = []
    for i in range(pitches.shape[1]):
            note = librosa.hz_to_note(pitch)
            notes.append(note)
    return notes[:20]

def extract_notes_from_audio(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    notes = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            note = librosa.hz_to_note(pitch)
            notes.append(note)
    return notes[:20]

