# moderatsiya-kartochek

Классификация изображений на предмет курения. Решение для соревнования OZON E-CUP.

## Prerequisites
- Docker (для работы с контейнером)

или для работы вне Docker:
- Python (>=3.10)
- Poetry (>=1.8.3)

##  Сборка и запуск проекта

### Через Docker
```shell
# Сборка образа
docker build . -t moderatsiya_kartochek
# Запуск контейнера для создания сабмита.
docker run -it --network none --shm-size 2G --name moderatsiya_kartochek -v "$(pwd)/data:/app/data" moderatsiya_kartochek python make_submission.py
# Вариант с аргументами
docker run -it --network none --shm-size 2G --name moderatsiya_kartochek -v "$(pwd)/data:/app/data" moderatsiya_kartochek make_submission data/test data
```


### Через Poetry
```shell
# Установка зависимостей.
poetry install --without dev,train
# Запуск скрипта для создания сабмита.
poetry run make_submission path/to/images path/to/save_dir
```

### Через Python-пакет
После сборки будет готов пакет в формате ```.wheel``` в папке ```dist```. Пакет далее можно установить
через менеджер пакетов ```pip```:
```shell
# Сборка пакета.
poetry install --without dev,train
poetry build --format wheel
# Установка пакета.
python -m venv venv 
source venv/bin/activate # Для Windows другая команда: venv/Scripts/activate
pip install dist/moderatsiya_kartochek-0.1.0-py3-none-any.whl
# Запуск скрипта для создания сабмита.
make_submission path/to/images path/to/save_dir
```

## Обучение модели
```shell
# Сборка образа.
docker build -t train_moderatsiya_kartochek -f Dockerfile-train .
# Запуск среды с ноутбуками.
docker run --rm -it --name train_moderatsiya_kartochek -p 8889:8889 --gpus all -v "$(pwd)/data:/app/data" -v "$(pwd)/notebooks:/app/notebooks" train_moderatsiya_kartochek
```

## Доступные скрипты
```shell
Usage: make_submission [OPTIONS] DATA_DIR SAVE_DIR

  Сабмит результата в SAVE_DIR на основе изображений из DATA_DIR.

Options:
  --threshold FLOAT  Пороговое значение.
  --help             Show this message and exit.
```