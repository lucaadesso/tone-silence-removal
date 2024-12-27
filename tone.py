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

def main(input_video, output_file, freq):
    # Estrai l'audio dal video
    video = VideoFileClip(input_video)
    audio =  AudioSegment.from_file(input_video)
    print("estratto audio dal video")

    # Trova i segmenti con toni sinusoidali a 1000 Hz
    tone_segments = find_tone_segments(audio, target_freq=freq, threshold=1e6)
    silence = [] # find_silence(audio)
    tone_segments += silence
    merged = sorted(set(tone_segments))
    merged = [(start /1000, stop /1000 ) for start , stop in merged]
    segmenti_attivi = []
    inizio = 0

    for start, stop in merged:
	# Aggiungi il segmento attivo prima dell'intervallo di silenzio
        if inizio < start:
            clip = video.subclip(inizio, start)
            clip = clip.audio_fadein(0.1).audio_fadeout(0.1)
            segmenti_attivi.append(clip)
        inizio = stop
    print(merged)
    print(inizio)
    print(video.duration)
    # Aggiungi l'ultimo segmento se esiste qualcosa dopo l'ultimo intervallo silenzioso
#    if 1 <= inizio < video.duration:
    if inizio < video.duration and abs(video.duration - inizio) >= 1:
        # clip = video.subclip(inizio, video.duration)
        clip = video.subclip(inizio, min(video.duration, inizio))
        clip = clip.audio_fadein(0.1).audio_fadeout(0.1)
        segmenti_attivi.append(clip)
    # Concatenazione dei segmenti attivi
    print("concateno i segmenti attivi")
    video_finale = concatenate_videoclips(segmenti_attivi)
    video_finale.write_videofile(output_file, codec="h264_nvenc", audio_codec="aac")
    video_finale.close()
    video.reader.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rileva toni sinusoidali a 1000 Hz in un video.")
    parser.add_argument("input_video", default="", help="Percorso al file video di input.")
    parser.add_argument("output_file", help="Percorso al file di output per salvare gli intervalli.")
    parser.add_argument("freq", default=5000, type=int)
    args = parser.parse_args()

    main(args.input_video, args.output_file, args.freq)
