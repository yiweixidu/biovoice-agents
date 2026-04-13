# ── BioVoice-Agents Dockerfile ────────────────────────────────────────────────
# Multi-stage: builder installs deps, final image copies site-packages only.
# Keeps the final image ~600MB instead of ~1.2GB.

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed only at build time (PDF conversion, LibreOffice for PPT→PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source code (layer cache)
COPY pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir --prefix=/install \
        "." \
        --find-links . \
    || pip install --no-cache-dir --prefix=/install \
        biopython langchain langchain-openai langchain-community \
        langchain-text-splitters langchain-huggingface chromadb openai \
        "pydantic>=2.0" pydantic-settings python-dotenv requests \
        "pdfplumber>=0.10.0" python-pptx edge-tts "moviepy==1.0.3" \
        pdf2image gradio fastapi uvicorn click pandas matplotlib seaborn \
        "networkx>=3.0" "python-docx>=1.1"

# ── Stage 2: final ────────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="BioVoice-Agents"
LABEL org.opencontainers.image.description="Multi-agent biomedical literature analysis"
LABEL org.opencontainers.image.source="https://github.com/your-org/biovoice-agents"

# Runtime system deps
# - poppler-utils: pdf2image (PPT→video pipeline)
# - libgl1: matplotlib/OpenCV
# - ffmpeg: moviepy video generation
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1-mesa-glx \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source
COPY biovoice/ ./biovoice/
COPY core/      ./core/
COPY domain/    ./domain/
COPY app/       ./app/
COPY scripts/   ./scripts/

# Create data directories
RUN mkdir -p data/vector_db data/cache output templates

# Non-root user for security
RUN groupadd -r biovoice && useradd -r -g biovoice biovoice
RUN chown -R biovoice:biovoice /app
USER biovoice

# Expose Gradio / FastAPI port
EXPOSE 7860

# Default: launch the Gradio web UI
CMD ["python3", "-m", "biovoice.app.gradio_app"]
