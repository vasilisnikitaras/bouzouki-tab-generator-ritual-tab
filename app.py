import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

string_bases = {'Ντο': 48, 'Φα': 53, 'Λα': 57, 'Ρε': 62}
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_midi(freq):
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def note_to_midi(note):
    name = note[:-1]
    octave = int(note[-1])
    return note_names.index(name) + 12 * (octave + 1)

def midi_to_note(midi):
    name = note_names[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"

def midi_to_freq(midi):
    return round(440 * 2 ** ((midi - 69) / 12), 2)

def find_positions(midi):
    return [(s, midi - b) for s, b in string_bases.items() if 0 <= midi - b <= 12]

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

# 🔹 Streamlit UI
st.title("🎼 Τελετουργική Ταμπλατούρα για Τετράχορδο Μπουζούκι")

input_type = st.radio("Είσοδος:", ["Νότα", "Συχνότητα"])
if input_type == "Νότα":
    note = st.text_input("Δώσε νότα (π.χ. G#4):")
    if note:
        midi = note_to_midi(note)
        st.write(f"Συχνότητα: {midi_to_freq(midi)} Hz")
        st.write("Θέσεις:")
        for s, f in find_positions(midi):
            st.write(f"→ Χορδή: {s}, Τάστο: {f}")
        plot_positions(midi)
elif input_type == "Συχνότητα":
    freq = st.number_input("Δώσε συχνότητα (Hz):", min_value=50.0, max_value=2000.0)
    if freq:
        midi = freq_to_midi(freq)
        note = midi_to_note(midi)
        st.write(f"Νότα: {note}")
        st.write("Θέσεις:")
        for s, f in find_positions(midi):
            st.write(f"→ Χορδή: {s}, Τάστο: {f}")
        plot_positions(midi)

st.subheader("🎵 Ταμπλατούρα με διάρκεια")
note_input = st.text_area("Λίστα νοτών με διάρκεια (π.χ. D4,0.5; F#4,1.0; A4,0.25)")
if note_input:
    entries = [tuple(x.strip().split(',')) for x in note_input.split(';')]
    parsed = [(n.strip(), float(d)) for n, d in entries]
    tab = tab_from_notes(parsed)
    for t in tab:
        st.write(f"{t['Νότα']} → Χορδή: {t['Χορδή']}, Τάστο: {t['Τάστο']}, Διάρκεια: {t['Διάρκεια']}")

st.subheader("🔮 AI πρόταση για ελάχιστη μετακίνηση")
if note_input:
    last_fret = None
    for t in tab:
        if t['Τάστο'] != '—':
            if last_fret is not None and abs(t['Τάστο'] - last_fret) > 5:
                st.write(f"👉 Εναλλακτική: Παίξε {t['Νότα']} σε άλλη χορδή για λιγότερη μετακίνηση.")
            last_fret = t['Τάστο']
