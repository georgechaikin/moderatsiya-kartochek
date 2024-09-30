FROM python:3.10-slim

RUN apt-get update && \
    apt-get install dos2unix && \
    apt-get clean

WORKDIR /app
VOLUME /app/data
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3","/app/make_submission.py"]

COPY . /app

RUN dos2unix /app/entrypoint.sh

RUN pip install poetry==1.8.3 && \
    poetry config virtualenvs.create false && \
    poetry install --without dev,train && \
    chmod +x /app/entrypoint.sh /app/make_submission.py
