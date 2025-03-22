import argparse
import cv2
from pydub import AudioSegment, silence
from scipy.fftpack import fft
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips
from tqdm import tqdm

def find_tone_segments(audio, target_freq=5000, threshold=0.1, chunk_size=100):
    """
    Trova segmenti audio con toni a una frequenza target.

    :param audio_path: Path del file audio. No
    :param target_freq: Frequenza target (in Hz). In Audacity tono da 10000hz su traccia stereo
    :param threshold: Valore soglia per rilevare la presenza della frequenza.
    :param chunk_size: Durata del chunk audio in millisecondi da analizzare.
    :return: Lista di tuple con gli intervalli temporali (start, end) in millisecondi.
    """
    duration_ms = len(audio)

    tone_segments = []
    current_start = None

    for start_ms in tqdm(range(0, duration_ms, chunk_size), desc="Analizzando audio"):
        chunk = audio[start_ms:start_ms + chunk_size]
        samples = np.array(chunk.get_array_of_samples())
        fft_result = np.abs(fft(samples))
        freqs = np.fft.fftfreq(len(fft_result), d=1 / chunk.frame_rate)

        # Trova l'indice della frequenza target
        idx = np.where((freqs >= target_freq - 5) & (freqs <= target_freq + 5))[0]

        # Controlla se la frequenza target supera la soglia
        if idx.size > 0 and np.max(fft_result[idx]) > threshold:
            if current_start is None:
                current_start = start_ms
        else:
            if current_start is not None:
                tone_segments.append((current_start, start_ms))
                current_start = None

    # Aggiungi l'ultimo segmento se ancora aperto
    if current_start is not None:
        tone_segments.append((current_start, duration_ms))

    return tone_segments

def find_silence(audio, silence_len=1250, silence_thresh=-80):
    """
    Trova segmenti di silenzio in un audio.

    :param audio: Un oggetto AudioSegment di pydub.
    :param silence_len: Lunghezza minima di silenzio in millisecondi.
    :param silence_thresh: Soglia di silenzio in dB.
    :return: Lista di intervalli [start, stop] che rappresentano i segmenti di silenzio.
    """
    with tqdm(total=len(audio), desc="Analisi del silenzio", unit="ms") as pbar:
        sil = silence.detect_silence(
            audio,
            min_silence_len=silence_len,
            silence_thresh=silence_thresh,
            seek_step=10  # Controlla ogni 10ms per migliorare la velocità
        )
        pbar.update(len(audio))

    # Aggiungi l'inizio del file (se necessario)
    dead_time = [] #[(0, sil[0][0])] if sil and sil[0][0] > 0 else []

    offset = 100

    # Aggiungi tutti i segmenti di silenzio trovati
    for start, stop in sil:
        dead_time.append((start + offset, stop - offset))  # Piccolo offset per evitare problemi

    return dead_time

def main(input_file, output_file, freq):
    # Determina se il file è un video o un audio
    try:
        video = VideoFileClip(input_file)
        is_video = True
        audio = AudioSegment.from_file(input_file)
    except Exception:
        is_video = False
        audio = AudioSegment.from_file(input_file)
    
    # Trova i segmenti con toni sinusoidali e silenzi
    tone_segments = find_tone_segments(audio, target_freq=freq, threshold=1e6)
    silence_segments = [] # find_silence(audio)
    all_segments = sorted(set(tone_segments + silence_segments))
    
    # Elabora il file in base al tipo
    if is_video:
        process_video(video, all_segments, output_file)
    else:
        process_audio(audio, all_segments, output_file)

def process_video(video, segments, output_file):
    """Processa un file video rimuovendo segmenti specifici."""
    segments = [(start / 1000, stop / 1000) for start, stop in segments]  # Converti in secondi
    active_segments = []
    inizio = 0

    for start, stop in segments:
        # Aggiungi il segmento attivo prima dell'intervallo di silenzio
        if inizio < start:
            clip = video.subclip(inizio, start)
            clip = clip.audio_fadein(0.1).audio_fadeout(0.1)
            active_segments.append(clip)
        inizio = stop
    
    # Aggiungi l'ultimo segmento
    if inizio < video.duration:
        clip = video.subclip(inizio, video.duration)
        clip = clip.audio_fadein(0.1).audio_fadeout(0.1)
        active_segments.append(clip)
    
    # Concatenazione e salvataggio del video
    final_video = concatenate_videoclips(active_segments)
    final_video.write_videofile(output_file, codec="h264_nvenc", audio_codec="aac")
    final_video.close()
    video.reader.close()

def process_audio(audio, segments, output_file):
    """Processa un file audio rimuovendo segmenti specifici."""
    output_audio = AudioSegment.silent(duration=0)  # Inizializza con un segmento vuoto
    last_end = 0

    for start, stop in segments:
        if last_end < start:
            output_audio += audio[last_end:start]  # Aggiungi la parte attiva
        last_end = stop
    
    if last_end < len(audio):
        output_audio += audio[last_end:]  # Aggiungi l'ultimo segmento
    
    # Salva l'audio elaborato
    output_audio.export(output_file, format="mp3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rimuove toni e silenzi da video o audio.")
    parser.add_argument("input_file", help="Percorso al file video o audio di input.")
    parser.add_argument("output_file", help="Percorso al file di output.")
    parser.add_argument("freq", default=5000, type=int, help="Frequenza target da rimuovere.")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.freq)
