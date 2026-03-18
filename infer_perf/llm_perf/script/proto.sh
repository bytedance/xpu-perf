#!/bin/bash

python3 -m grpc_tools.protoc -I./llm_perf --python_out=./llm_perf --grpc_python_out=./llm_perf ./llm_perf/server.proto