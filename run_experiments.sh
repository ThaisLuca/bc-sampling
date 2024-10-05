# !/bin/bash

python run.py --data-dir="datasets/imdb" --log-dir="logs/imdb" --target="workedunder"
python run.py --data-dir="datasets/uwcse" --log-dir="logs/uwcse" --target="advisedby"
python run.py --data-dir="datasets/cora" --log-dir="logs/cora" --target="samevenue"
python run.py --data-dir="datasets/yeast" --log-dir="logs/yeast" --target="proteinclass"
python run.py --data-dir="datasets/twitter" --log-dir="logs/twitter" --target="accounttype"
python run.py --data-dir="datasets/nell_sports" --log-dir="logs/nell_sports" --target="teamplayssport"
python run.py --data-dir="datasets/nell_finances" --log-dir="logs/nell_finances" --target="companyeconomicsector"


