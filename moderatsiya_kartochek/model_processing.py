import os

import numpy as np
import openvino as ov
from PIL import Image
from numpy.typing import NDArray
from transformers import BlipImageProcessor


def softmax(logits: NDArray[np.float32], axis: int = 1) -> NDArray[np.float32]:
    """Реализация softmax на основе функции из SciPy.

    См. https://github.com/scipy/scipy/blob/v1.14.1/scipy/special/_logsumexp.py#L141-L235

    Args:
        logits: Логиты.
        axis: Ось для вычисления значений.

    Returns:
        Предсказания на основе логитов.
    """
    logits_max = np.amax(logits, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(logits - logits_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def load_ov_model(model_path: str | os.PathLike) -> ov.runtime.ie_api.CompiledModel:
    """Загрузка модели на OpenVINO.

    Args:
        model_path: Файл для работы с моделью в формате .xml.

    Returns:
        Модель на OpenVINO.
    """
    core = ov.Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name="CPU")
    return compiled_model


def predict_class(
    img_path: str | os.PathLike,
    ov_model: ov.runtime.ie_api.CompiledModel,
    processor: BlipImageProcessor,
    threshold: float,
) -> int:
    """Предсказание класса.

    Args:
        img_path: Путь до изображения.
        ov_model: Модель на OpenVINO.
        processor: Обработчик изображения в аргумент для модели.
        threshold: Пороговое значение для вероятности.

    Returns:
        Номер класса.
    """
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="np")
    pixel_values = inputs["pixel_values"]
    logits = ov_model(pixel_values)[ov_model.output(0)]
    preds = softmax(logits, -1)[0]
    class_id = np.where(preds[1] > threshold, 1, 0).item()
    return class_id
