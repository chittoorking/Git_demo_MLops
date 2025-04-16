
FROM python:3.10
WORKDIR /app
COPY main.py /app/main.py
COPY requirements.txt /app/
COPY *.pkl /app/
RUN pip install -r requirements.txt
EXPOSE 5000/tcp

ENTRYPOINT [ "python", "main.py" ]