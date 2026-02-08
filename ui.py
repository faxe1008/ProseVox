import os
import json
import shutil
import re
import torch
import soundfile as sf
import gradio as gr
import pyloudnorm as pyln
import numpy as np

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

# =========================
# Global cache (EPUB parsed once)
# =========================
EPUB_CACHE = {
    "path": None,
    "paragraphs": None,
}

OUTPUT_DIR = "audiobook_output"
META_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# EPUB parsing
# =========================
def extract_paragraphs_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    paragraphs = []

    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            content = item.get_body_content()
            if not content:
                continue

            html = content.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")

            for p in soup.find_all("p"):
                text = p.get_text().strip()
                if text:
                    paragraphs.append(text)

    return paragraphs

# =========================
# Paragraph → sentence-safe segments
# =========================
def split_into_segments(paragraphs, max_chars=1500, min_sentences=3):
    """
    Strong guarantees:
      - Segments ALWAYS contain at least `min_sentences` sentences,
        unless fewer sentences remain in total.
      - Paragraph boundaries are treated as soft hints, not hard barriers.
      - Sentence boundaries are never broken.
      - max_chars is respected when possible, but min_sentences wins.
    """

    # 1. Flatten all paragraphs into a single sentence stream
    sentences = []
    for para in paragraphs:
        parts = re.split(r'(?<=[.!?])\s+', para)
        sentences.extend(s.strip() for s in parts if s.strip())

    segments = []
    i = 0
    n = len(sentences)

    while i < n:
        current = []
        char_count = 0

        # 2. First: force min_sentences
        while i < n and len(current) < min_sentences:
            s = sentences[i]
            current.append(s)
            char_count += len(s) + 1
            i += 1

        # 3. Then: grow until max_chars (soft limit)
        while i < n:
            s = sentences[i]
            if char_count + len(s) + 1 > max_chars:
                break
            current.append(s)
            char_count += len(s) + 1
            i += 1

        segments.append(" ".join(current))

    return segments


# =========================
# Loudness normalization (LUFS)
# =========================
def normalize_loudness(wav, sr, target_lufs=-16.0, peak_limit=0.98):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)

    normalized = pyln.normalize.loudness(
        wav,
        loudness,
        target_lufs
    )

    peak = np.max(np.abs(normalized))
    if peak > peak_limit:
        normalized *= peak_limit / peak

    return normalized.astype(wav.dtype)

# =========================
# Helpers
# =========================
def _get_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", None)

# =========================
# Preview trimming
# =========================
def preview_trimmed_text(epub_file, skip_start, skip_end):
    epub_path = _get_path(epub_file)
    if epub_path is None:
        return "No EPUB selected."

    if EPUB_CACHE["path"] != epub_path:
        EPUB_CACHE["paragraphs"] = extract_paragraphs_from_epub(epub_path)
        EPUB_CACHE["path"] = epub_path

    paragraphs = EPUB_CACHE["paragraphs"]
    total = len(paragraphs)

    skip_start = int(skip_start or 0)
    skip_end = int(skip_end or 0)

    if skip_start + skip_end >= total:
        return "⚠️ Trimming removes all content."

    kept = paragraphs[skip_start: total - skip_end]

    out = []
    out.append(f"Total paragraphs: {total}")
    out.append(f"Trimmed start: {skip_start}")
    out.append(f"Trimmed end: {skip_end}")
    out.append("\n--- NARRATED TEXT ---\n")

    for p in kept:
        out.append(p)

    return "\n\n".join(out)

# =========================
# Generated segment helpers
# =========================
def list_generated_segment_files():
    if not os.path.isdir(OUTPUT_DIR):
        return []
    return sorted(
        fn for fn in os.listdir(OUTPUT_DIR)
        if fn.lower().endswith(".wav") and fn.startswith("audiobook_")
    )

def list_generated_segments_for_dropdown():
    files = list_generated_segment_files()
    if not files:
        return gr.update(choices=[], value=None)
    return gr.update(choices=files, value=files[0])

# =========================
# Audiobook generation (streaming)
# =========================
def generate_audiobook(epub_file, skip_start, skip_end, ref_audio, ref_text):
    epub_path = _get_path(epub_file)
    if epub_path is None:
        yield ("❌ No EPUB provided.", "", list_generated_segments_for_dropdown())
        return

    ref_audio_path = _get_path(ref_audio)
    if ref_audio_path is None or not ref_text or not ref_text.strip():
        yield ("❌ Reference audio and transcript are required.", "", list_generated_segments_for_dropdown())
        return

    local_epub = os.path.join(OUTPUT_DIR, "book.epub")
    shutil.copyfile(epub_path, local_epub)

    skip_start = int(skip_start or 0)
    skip_end = int(skip_end or 0)

    if os.path.exists(META_FILE):
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        segments = meta["segments"]
    else:
        paragraphs = extract_paragraphs_from_epub(local_epub)
        trimmed = paragraphs[skip_start: len(paragraphs) - skip_end]
        texts = split_into_segments(trimmed)

        segments = []
        for i, text in enumerate(texts, 1):
            segments.append({
                "index": i,
                "text": text,
                "file": f"audiobook_{i:06}.wav"
            })

        meta = {
            "skip_start": skip_start,
            "skip_end": skip_end,
            "segments": segments
        }

        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # =========================
    # SAFE reference audio loading (FIX)
    # =========================
    try:
        ref_wav, ref_sr = sf.read(ref_audio_path)

        ref_wav = np.asarray(ref_wav, dtype=np.float32)

        if ref_wav.ndim > 1:
            ref_wav = ref_wav.mean(axis=1)

        ref_wav = ref_wav.copy()  # REQUIRED for Qwen

    except Exception as e:
        yield (f"❌ Failed to read reference audio: {e}", "", list_generated_segments_for_dropdown())
        return

    # =========================
    # Model setup
    # =========================
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # much faster + less VRAM
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    from qwen_tts import Qwen3TTSModel

    try:
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
    except Exception as e:
        yield (f"❌ Failed to load model: {e}", "", list_generated_segments_for_dropdown())
        return

    try:
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=(ref_wav, ref_sr),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
    except Exception as e:
        yield (f"❌ Failed to create voice clone prompt: {e}", "", list_generated_segments_for_dropdown())
        return

    total_segments = len(segments)
    width = len(str(total_segments))
    generated = 0

    for seg in segments:
        out_path = os.path.join(OUTPUT_DIR, seg["file"])
        if os.path.exists(out_path):
            yield (
                f"Skipping {seg['index']:0{width}d}/{total_segments}",
                seg["text"],
                None
            )
            continue

        try:
            with torch.inference_mode():
                wavs, sr = model.generate_voice_clone(
                    text=seg["text"],
                    language="German",
                    voice_clone_prompt=prompt_items,
                )
        except Exception:
            try:
                model.to("cpu")
                model.float()
                wavs, sr = model.generate_voice_clone(
                    text=seg["text"],
                    language="German",
                    voice_clone_prompt=prompt_items,
                )
            except Exception as final_exc:
                yield (f"❌ Generation failed for segment {seg['index']}: {final_exc}", seg["text"], None)
                return

        audio = wavs[0]
        try:
            audio = normalize_loudness(audio, sr)
        except Exception:
            pass

        sf.write(out_path, audio, sr)
        generated += 1

        yield (
            f"Generating {seg['index']:0{width}d}/{total_segments} (new: {generated})",
            seg["text"],
            None
        )

    yield (
        f"✅ Done. Generated {generated} new segments. Total: {total_segments}.",
        "",
        list_generated_segments_for_dropdown()
    )

# =========================
# Playback
# =========================
def play_selected_segment(selected_filename):
    if not selected_filename:
        return None
    path = os.path.join(OUTPUT_DIR, selected_filename)
    return path if os.path.exists(path) else None

# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 📘 EPUB → Audiobook (Qwen3-TTS Voice Cloning)")

    with gr.Row():
        with gr.Column(scale=2):
            epub_in = gr.File(label="Upload EPUB", file_types=[".epub"], type="filepath")

            with gr.Row():
                skip_start = gr.Number(label="Trim start (paragraphs)", value=0, precision=0)
                skip_end = gr.Number(label="Trim end (paragraphs)", value=0, precision=0)

            preview_btn = gr.Button("📖 Preview Trimmed Text")
            preview_box = gr.Textbox(lines=10, interactive=False)

            preview_btn.click(
                preview_trimmed_text,
                inputs=[epub_in, skip_start, skip_end],
                outputs=preview_box
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎙 Voice Cloning")
            ref_audio = gr.File(label="Reference voice (.wav)", type="filepath")
            ref_text = gr.Textbox(label="Transcript", lines=4)

            gen_btn = gr.Button("🎧 Generate Audiobook", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, lines=1)
            current_seg_box = gr.Textbox(label="Current segment", interactive=False, lines=3)

            gr.Markdown("### ▶️ Listen to generated segments")
            segments_dropdown = gr.Dropdown(
                label="Generated segments",
                choices=list_generated_segment_files(),
                value=None
            )

            audio_player = gr.Audio(label="Audio player", interactive=False, autoplay=True)

            segments_dropdown.change(
                fn=play_selected_segment,
                inputs=segments_dropdown,
                outputs=audio_player
            )

            gen_btn.click(
                generate_audiobook,
                inputs=[epub_in, skip_start, skip_end, ref_audio, ref_text],
                outputs=[status_box, current_seg_box, segments_dropdown],
            )

    gr.HTML("""
    <script>
    document.addEventListener("ended", function(e) {
        if (e.target.tagName === "AUDIO") {
            const dropdown = document.querySelector("select");
            if (!dropdown) return;
            const next = dropdown.selectedIndex + 1;
            if (next < dropdown.options.length) {
                dropdown.selectedIndex = next;
                dropdown.dispatchEvent(new Event("change", { bubbles: true }));
            }
        }
    }, true);
    </script>
    """)

demo.launch()
