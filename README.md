
## Regression laba
код лежит в https://github.com/Borshick02/HH_salary.git
Проект лежит в hh_salary_model/
установка зависимостей
```bash
cd hh_salary_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

обучение модели
```bash
python scripts/train.py ../x_data.npy ../y_data.npy
```
После этого веса сохраняются в:

hh_salary_model/resources/model.npz

Вывести список зарплат в stdout (в рублях, float)
```bash
python app ../x_data.npy
```
