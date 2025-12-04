FROM python:3.11-slim
WORKDIR /app

# system deps for some packages (faiss may require build; for dev we use faiss-cpu wheel)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 2001

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "2001"]
