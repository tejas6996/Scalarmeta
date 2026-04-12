# ── Base image: lightweight Python 3.11 ──────────────────────────────────────
FROM python:3.11-slim

# Non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY --chown=user . .

# ── Runtime environment variables (override in HF Spaces secrets) ────────────
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""
ENV PORT=7860
ENV PYTHONPATH="/app"

# ── Expose port 7860 (Hugging Face Spaces default) ───────────────────────────
EXPOSE 7860

# ── Start the FastAPI server ─────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
