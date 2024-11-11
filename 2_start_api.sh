cd "$(dirname "$0")/api"

. ~/venvs/evs/bin/activate

pip install -r requirements.txt
python api.py