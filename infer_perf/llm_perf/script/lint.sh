#!/bin/bash

python3 -m isort llm_perf -s model_impl -s model_zoo
python3 -m black llm_perf --extend-exclude "model_impl|model_zoo"