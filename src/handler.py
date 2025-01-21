""" Example handler file. """

import runpod
import os
import argparse

from pulid import app_flux

def get_args_from_env():
    # Устанавливаем значения по умолчанию
    defaults = {
        "version": "v0.9.1",
        "name": "flux-dev",
        "device": "cuda",
        "offload": True,
        "aggressive_offload": False,
        "fp8": True,
        "onnx_provider": "gpu",
        "dev": False,
        "pretrained_model": None
    }

    # Получаем значения из переменных окружения
    args = {
        "version": os.getenv("FLUX_VERSION", defaults["version"]),
        "name": os.getenv("FLUX_NAME", defaults["name"]),
        "device": os.getenv("FLUX_DEVICE", defaults["device"]),
        "offload": os.getenv("FLUX_OFFLOAD", str(defaults["offload"])).lower() == "true",
        "aggressive_offload": os.getenv("FLUX_AGGRESSIVE_OFFLOAD", str(defaults["aggressive_offload"])).lower() == "true",
        "fp8": os.getenv("FLUX_FP8", str(defaults["fp8"])).lower() == "true",
        "onnx_provider": os.getenv("FLUX_ONNX_PROVIDER", defaults["onnx_provider"]),
        "dev": os.getenv("FLUX_DEV", str(defaults["dev"])).lower() == "true",
        "pretrained_model": os.getenv("FLUX_PRETRAINED_MODEL", defaults["pretrained_model"]),
    }

    # Если агрессивный оффлоад включен, включаем обычный оффлоад автоматически
    if args["aggressive_offload"]:
        args["offload"] = True

    # Преобразуем словарь в объект Namespace для совместимости с argparse
    return argparse.Namespace(**args)

def init_generator():
    args = get_args_from_env()
    
    return app_flux.FluxGenerator(
        model_name=args.name,
        device=args.device,
        offload=args.offload,
        aggressive_offload=args.aggressive_offload,
        args=args
    )

generator = init_generator()

def handler(job):
    job_input = job['input']

    # Подготовка аргументов для генерации изображения
    width = job_input.get('width', 896)  # Значение по умолчанию 896
    height = job_input.get('height', 1152)  # Значение по умолчанию 1152
    num_steps = job_input.get('num_steps', 20)  # Значение по умолчанию 20
    start_step = job_input.get('start_step', 0)  # Значение по умолчанию 0
    guidance = job_input.get('guidance', 4)  # Значение по умолчанию 4
    seed = job_input.get('seed', -1)  # Значение по умолчанию -1
    prompt = job_input.get('prompt', "portrait, color, cinematic")  # Значение по умолчанию
    neg_prompt = job_input.get('neg_prompt', "bad quality, worst quality, text, signature, watermark, extra limbs")
    true_cfg = job_input.get('true_cfg', 1.0)  # Значение по умолчанию 1.0
    timestep_to_start_cfg = job_input.get('timestep_to_start_cfg', 1)  # Значение по умолчанию 1
    max_sequence_length = job_input.get('max_sequence_length', 128)  # Значение по умолчанию 128
    id_weight = job_input.get('id_weight', 1.0)  # Значение по умолчанию 1.0

    # Загрузка id_image, если передан URL
    id_image = None
    id_image_url = job_input.get('id_image')
    if id_image_url:
        try:
            response = requests.get(id_image_url)
            response.raise_for_status()
            id_image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Ошибка при загрузке id_image: {e}")
            id_image = None

    # Генерация изображения
    generated_image, used_seed, debug_images = generator.generate_image(
        width=width,
        height=height,
        num_steps=num_steps,
        start_step=start_step,
        guidance=guidance,
        seed=seed,
        prompt=prompt,
        id_image=id_image,
        id_weight=id_weight,
        neg_prompt=neg_prompt,
        true_cfg=true_cfg,
        timestep_to_start_cfg=timestep_to_start_cfg,
        max_sequence_length=max_sequence_length,
    )

    # Сохранение результата в виде URL (например, в локальной файловой системе или облаке)
    result_path = f"/path/to/save/generated_image_{used_seed}.png"
    generated_image.save(result_path)

    # Вернуть URL результата
    return f"http://yourserver.com/results/{used_seed}.png"


runpod.serverless.start({"handler": handler})
