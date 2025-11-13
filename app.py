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

st.set_page_config(page_title="Τελετουργική Ταμπλατούρα", page_icon="🎼")
st.title("🎼 Τελετουργική Ταμπλατούρα για Τετράχορδο Μπουζούκι")
st.markdown("Καλώς ήρθες στην τελετουργική εφαρμογή για μετατροπή νοτών, συχνοτήτων και τραγουδιών σε ταμπλατούρα για τετράχορδο μπουζούκι.")

string_bases = {'Ντο': 48, 'Φα': 53, 'Λα': 57, 'Ρε': 62}
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
greek_names = {'C': 'Ντο', 'C#': 'Ντο#', 'D': 'Ρε', 'D#': 'Ρε#', 'E': 'Μι', 'F': 'Φα', 'F#': 'Φα#', 'G': 'Σολ', 'G#': 'Σολ#', 'A': 'Λα', 'A#': 'Λα#', 'B': 'Σι'}

def freq_to_midi(freq): return int(round(69 + 12 * np.log2(freq / 440.0)))
def midi_to_freq(midi): return round(440 * 2 ** ((midi - 69) / 12), 2)
def midi_to_note(midi):
    name = note_names[midi % 12]
    octave = midi // 12 - 1
    greek = greek_names.get(name, name)
    return f"{name}{octave} / {greek} / MIDI:{midi} / {midi_to_freq(midi)}Hz"

def note_to_midi(note):
    match = re.match(r'^([A-G]#?|[A-G]b?)(-?\d+)$', note.strip())
    if not match: raise ValueError(f"Μη έγκυρη νότα: {note}")
    name, octave = match.groups()
    return note_names.index(name) + 12 * (int(octave) + 1)

def find_positions(midi):
    return [(s, midi - b) for s, b in string_bases.items() if 0 <= midi - b <= 12]

def plot_positions(midi):
    positions = find_positions(midi)
    fig, ax = plt.subplots(figsize=(10, 4))
    strings = list(string_bases.keys())
    ax.set_yticks(range(len(strings)))
    ax.set_yticklabels(strings)
    ax.set_xticks(range(13))
    ax.grid(True)
    for s, f in positions:
        y = strings.index(s)
        ax.plot(f, y, 'ro', markersize=12)
        ax.text(f, y + 0.2, midi_to_note(midi), ha='center')
    st.pyplot(fig)

def tab_from_notes(note_list):
    tab = []
    for note, dur in note_list:
        midi = note_to_midi(note)
        pos = find_positions(midi)
        if pos:
            s, f = pos[0]
            tab.append({'Νότα': midi_to_note(midi), 'Χορδή': s, 'Τάστο': f, 'Διάρκεια': dur})
        else:
            tab.append({'Νότα': midi_to_note(midi), 'Χορδή': '—', 'Τάστο': '—', 'Διάρκεια': dur})
    return tab


def generate_pdf(tab):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="🎼 Τελετουργική Ταμπλατούρα", ln=True, align='C')
    for t in tab:
        line = f"{t['Νότα']} → Χορδή: {t['Χορδή']}, Τάστο: {t['Τάστο']}, Διάρκεια: {t['Διάρκεια']}"
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output("tab.pdf")
    return "tab.pdf"

def export_midi(tab, filename="output.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for t in tab:
        try:
            midi = note_to_midi(t['Νότα'].split()[0])
            duration = int(t['Διάρκεια'] * 480)
            track.append(Message('note_on', note=midi, velocity=64, time=0))
            track.append(Message('note_off', note=midi, velocity=64, time=duration))
        except: continue
    mid.save(filename)
    return filename

def plot_spectrum(file_path):
    y, sr = librosa.load(file_path)
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    ax.set_title("📈 Φασματική Ανάλυση")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s)',
        'ffmpeg_location': r'C:\Users\Admin\Downloads\ffmpeg\bin',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'audio.wav'

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


# 🎚️ Επιλογή είδους εισόδου
input_type = st.radio("📥 Επιλέξτε είδος εισόδου:", ["Νότα", "Συχνότητα", "Αρχείο Ήχου", "YouTube"])

if input_type == "Νότα":
    note_input = st.text_input("🎵 Εισάγετε νότα (π.χ. G#4):")
    if note_input:
        try:
            midi = note_to_midi(note_input)
            st.write(f"🎼 {midi_to_note(midi)}")
            plot_positions(midi)
        except Exception as e:
            st.error(f"⚠️ Σφάλμα: {e}")

elif input_type == "Συχνότητα":
    freq_input = st.number_input("📡 Εισάγετε συχνότητα (Hz):", min_value=20.0, max_value=2000.0)
    if freq_input:
        midi = freq_to_midi(freq_input)
        st.write(f"🎼 {midi_to_note(midi)}")
        plot_positions(midi)

elif input_type == "Αρχείο Ήχου":
    uploaded_file = st.file_uploader("🎙️ Ανεβάστε αρχείο .wav", type=["wav"])
    if uploaded_file:
        with open("uploaded.wav", "wb") as f:
            f.write(uploaded_file.read())
        notes = extract_notes_from_audio("uploaded.wav")
        st.write("🎵 Εξαγόμενες Νότες:", notes)
        tab = tab_from_notes([(n, 1) for n in notes])
        st.subheader("📜 Ταμπλατούρα")
        for t in tab:
            st.write(f"{t['Νότα']} → Χορδή: {t['Χορδή']}, Τάστο: {t['Τάστο']}, Διάρκεια: {t['Διάρκεια']}")
        plot_positions(note_to_midi(notes[0]))

elif input_type == "YouTube":
    yt_link = st.text_input("📺 Εισάγετε σύνδεσμο YouTube:")
    if yt_link:
        audio_path = download_youtube_audio(yt_link)
        plot_spectrum(audio_path)
        notes = extract_notes_from_audio(audio_path)
        st.write("🎵 Εξαγόμενες Νότες:", notes)
        tab = tab_from_notes([(n, 1) for n in notes])
        st.subheader("📜 Ταμπλατούρα")
        for t in tab:
            st.write(f"{t['Νότα']} → Χορδή: {t['Χορδή']}, Τάστο: {t['Τάστο']}, Διάρκεια: {t['Διάρκεια']}")
        plot_positions(note_to_midi(notes[0]))

# 📄 Εξαγωγή PDF
if st.button("📄 Εξαγωγή PDF Ταμπλατούρας"):
    if 'tab' in locals() and tab:
        pdf_path = generate_pdf(tab)
        st.success("✅ Το PDF δημιουργήθηκε.")
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Κατέβασε το PDF", f, file_name="tab.pdf")
    else:
        st.error("⚠️ Δεν υπάρχει διαθέσιμη ταμπλατούρα για εξαγωγή.")

# 🎼 Εξαγωγή MIDI
if st.button("🎼 Εξαγωγή MIDI"):
    if 'tab' in locals() and tab:
        midi_path = export_midi(tab)
        st.success("✅ Το MIDI δημιουργήθηκε.")
        with open(midi_path, "rb") as f:
            st.download_button("📥 Κατέβασε το MIDI", f, file_name="output.mid")
    else:
        st.error("⚠️ Δεν υπάρχει διαθέσιμη ταμπλατούρα για εξαγωγή.")
