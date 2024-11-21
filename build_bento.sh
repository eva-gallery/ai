#!/bin/bash

bentoml build
bentoml containerize evagallery_ai_api:latest --opt progress=plain
