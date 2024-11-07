#! /bin/bash

poetry lock --no-update
poetry install
poetry run BENTOML_CONFIG=bento_config.yaml bentoml serve ai_api.captioning_model:captioning_model