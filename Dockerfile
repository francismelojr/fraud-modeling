FROM python:3.11.5-slim

RUN apt-get update && apt-get -y install libgomp1

RUN pip install poetry

WORKDIR /src

COPY ["pyproject.toml", "poetry.lock", "./"]

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev

COPY ["src/model/predict.py", "src/model/model_lgbm.bin", "./"]

EXPOSE 9696

ENTRYPOINT  ["python", "predict.py"]