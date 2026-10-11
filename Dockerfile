# exact base and exact dependency versions (requirements.lock, `make lock`):
# a rebuild installs what was tested, not whatever is newest that day
FROM docker.io/library/python:3.12.15-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# lightgbm needs libgomp at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY db ./db

RUN useradd --create-home --uid 10001 malrec && chown -R malrec /app
USER malrec

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "malrec.api:app", "--host", "0.0.0.0", "--port", "8000"]
