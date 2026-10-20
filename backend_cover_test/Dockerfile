FROM python:3.12.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /srv/remicare
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt && python -m pip check
COPY app ./app
COPY models ./models
COPY docker ./docker
RUN useradd --uid 10001 --create-home remicare \
    && mkdir -p data && chown -R remicare:remicare /srv/remicare
USER remicare
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=12s --start-period=60s --retries=3 \
    CMD ["python", "docker/healthcheck.py"]
CMD ["python", "docker/start.py"]
