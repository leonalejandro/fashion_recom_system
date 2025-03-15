# Add a non-root user
#RUN addgroup --gid $GID app
#RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app
#USER app

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
FROM python:3.9-slim as base
ENV PYTHONPATH=$PYTHONPATH:/src/
ENV PATH=$PATH:/home/app/.local/bin/

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/src/

COPY ./ /src/
COPY data /src/

WORKDIR /src

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

FROM base as build
ENTRYPOINT ["python3", "/src/recomendation_service.py"]