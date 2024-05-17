FROM python:3.11.1-alpine3.17

RUN pip install --no-cache-dir \
    redis \
    loguru \
    pydantic==1.10.14 \
    flask

WORKDIR /src

COPY setup.py /src/setup.py
COPY kamerad /src/kamerad

RUN pip install .

ENTRYPOINT ["/src/kamerad/appserver.py", "-H", "0.0.0.0"]
