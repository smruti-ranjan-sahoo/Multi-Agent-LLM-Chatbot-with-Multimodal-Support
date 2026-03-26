from faster_whisper import WhisperModel

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

def speech_to_text(audio_path: str) -> str:
    segments, _ = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,          # 🔥 improves accuracy
        vad_filter=True       # 🔥 removes silence
    )

    text = " ".join([seg.text for seg in segments])
    return text.strip()