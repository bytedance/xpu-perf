#！bin/bash
if [ ! -d "general_perf/backends/CPU/venv" ];then
    virtualenv -p python3 general_perf/backends/CPU/venv
    source general_perf/backends/CPU/venv/bin/activate
    general_perf/backends/CPU/venv/bin/python3 -m pip install --upgrade pip  -q
    general_perf/backends/CPU/venv/bin/python3 -m pip install -r general_perf/backends/CPU/requirements.txt -q
else
    source general_perf/backends/CPU/venv/bin/activate
    general_perf/backends/CPU/venv/bin/python3 -m pip install -r general_perf/backends/CPU/requirements.txt -q
fi

python3 general_perf/backends/CPU/calculate_cpu_diff.py --task $1 --batch_size $2
