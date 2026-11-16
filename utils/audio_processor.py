
import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def normalize_audio(input_path: str) -> str:
    """Convert audio to mono, 16 kHz WAV."""

    output_path = os.path.splitext(input_path)[0] + "_normalized.wav"

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)

    audio.export(output_path, format="wav")

    return output_path


def download_youtube_audio(url: str) -> str:

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "cookiesfrombrowser": ("firefox",),

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],

        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(url, download=True)

        filename = ydl.prepare_filename(info)

        # FFmpegExtractAudio creates a WAV file.
        wav_path = os.path.splitext(filename)[0] + ".wav"

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(
            f"Downloaded WAV file not found: {wav_path}"
        )

    return normalize_audio(wav_path)


def convert_to_wav(input_path: str) -> str:

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    return normalize_audio(input_path)


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 8
) -> list[str]:

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        size_mb = os.path.getsize(chunk_path) / (1024 * 1024)

        print(
            f"Created chunk {i + 1}: {size_mb:.2f} MB"
        )

        if size_mb >= 24:
            raise ValueError(
                f"Audio chunk is too large ({size_mb:.2f} MB). "
                "Reduce chunk_minutes to 6 or 5."
            )

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list[str]:

    if source.startswith(("http://", "https://")):

        print("Detected YouTube URL. Downloading audio...")

        wav_path = download_youtube_audio(source)

    else:

        print("Detected local file. Converting to WAV...")

        wav_path = convert_to_wav(source)

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    print(f"Audio ready: {len(chunks)} chunk(s) created.")

    return chunks
