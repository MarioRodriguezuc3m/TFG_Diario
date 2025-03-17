FROM pypy:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/plots
VOLUME /app/plots
COPY src/ ./src/
CMD ["pypy3", "src/main.py"]