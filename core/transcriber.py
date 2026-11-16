
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv(
    "GROQ_WHISPER_MODEL",
    "whisper-large-v3-turbo"
)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def transcribe_chunk(
    chunk_path: str,
    translate: bool = False,
    language: str = "ur"
) -> str:

    if not os.path.isfile(chunk_path):
        raise FileNotFoundError(
            f"Audio file not found: {chunk_path}"
        )

    with open(chunk_path, "rb") as audio_file:

        if translate:
          
            result = client.audio.translations.create(
                file=(os.path.basename(chunk_path), audio_file),
                model="whisper-large-v3",
                temperature=0.0
            )

        else:
            
            result = client.audio.transcriptions.create(
                file=(os.path.basename(chunk_path), audio_file),
                model=GROQ_MODEL,
                language=language,
                temperature=0.0
            )

    return result.text.strip()


def transcribe_all(
    chunks: list[str],
    translate: bool = False,
    language: str = "ur"
) -> str:

    full_transcript = []

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk {i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            translate=translate,
            language=language
        )

        full_transcript.append(text)

    print("Transcription completed!")

    return "\n".join(full_transcript)
