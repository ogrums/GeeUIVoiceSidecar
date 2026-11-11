#!/usr/bin/env python3
"""Télécharge les huit expressions RAVDESS et prépare leur manifeste de test.

Par défaut, remotezip récupère seulement 24 WAV (trois par expression), sans
enregistrer l'archive de 199 Mio. --full télécharge l'archive complète puis
extrait tous les WAV. Sortie : PCM 16 kHz mono, licence CC BY-NC-SA 4.0.
"""
import argparse
import array
import csv
import io
from pathlib import Path
import sys
from urllib.request import urlopen
import wave
import zipfile

URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
RAW_LABELS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgusted", "08": "surprised",
}
ROBOT = {
    "neutral": "neutral", "calm": "neutral", "happy": "happy", "sad": "sad",
    "angry": "angry", "fearful": "fear", "disgusted": "neutral", "surprised": "surprise",
}
# Actrices (impair) et acteurs (pair), intensité normale puis forte. Le neutre
# n'existe qu'en intensité normale : on choisit donc une troisième personne.
SELECTED = {
    "neutral": [(1, "01"), (2, "01"), (3, "01")],
    "calm": [(1, "01"), (2, "01"), (1, "02")],
    "happy": [(1, "01"), (2, "01"), (1, "02")],
    "sad": [(1, "01"), (2, "01"), (1, "02")],
    "angry": [(1, "01"), (2, "01"), (1, "02")],
    "fearful": [(1, "01"), (2, "01"), (1, "02")],
    "disgusted": [(1, "01"), (2, "01"), (1, "02")],
    "surprised": [(1, "01"), (2, "01"), (1, "02")],
}
CODES = {raw: code for code, raw in RAW_LABELS.items()}


def selected_names():
    return [f"Actor_{actor:02d}/03-01-{CODES[label]}-{intensity}-01-01-{actor:02d}.wav"
            for label, choices in SELECTED.items() for actor, intensity in choices]


def wav_16k_mono(data):
    """Rééchantillonnage déterministe du WAV RAVDESS 48 kHz mono PCM 16 bits."""
    with wave.open(io.BytesIO(data), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Échantillon RAVDESS inattendu : mono PCM 16 bits requis")
        rate = source.getframerate()
        samples = array.array("h", source.readframes(source.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    output = array.array("h", (samples[min(len(samples) - 1, int(i * rate / 16000))]
                              for i in range(int(len(samples) * 16000 / rate))))
    if sys.byteorder != "little":
        output.byteswap()
    result = io.BytesIO()
    with wave.open(result, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(output.tobytes())
    return result.getvalue()


def extract(archive, names, output_dir):
    rows = []
    for name in names:
        parts = Path(name).name.split("-")
        raw = RAW_LABELS[parts[2]]
        destination = output_dir / Path(name).name
        destination.write_bytes(wav_16k_mono(archive.read(name)))
        rows.append({"file": destination.name, "ravdess_label": raw, "expected_emotion": ROBOT[raw]})
        print(f"{destination} : {destination.stat().st_size} octets")
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "ravdess_label", "expected_emotion"))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["file"]))
    print(f"{len(rows)} WAV ; licence CC BY-NC-SA 4.0 ; Livingstone & Russo, RAVDESS, Zenodo 1188976.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/samples/ravdess"))
    parser.add_argument("--full", action="store_true", help="Télécharger et extraire les 1440 WAV de l'archive complète")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.full:
        archive_path = args.output_dir.parent / "Audio_Speech_Actors_01-24.zip"
        if not archive_path.exists():
            print(f"Téléchargement de {URL} vers {archive_path}")
            with urlopen(URL, timeout=120) as response, archive_path.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        with zipfile.ZipFile(archive_path) as archive:
            extract(archive, [n for n in archive.namelist() if n.endswith(".wav")], args.output_dir)
    else:
        try:
            import remotezip
        except ImportError:
            parser.error("Installer remotezip : python3 -m pip install remotezip (ou utiliser --full)")
        with remotezip.RemoteZip(URL) as archive:
            extract(archive, selected_names(), args.output_dir)


if __name__ == "__main__":
    main()
