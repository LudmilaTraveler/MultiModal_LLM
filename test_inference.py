import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

def run_test_inference():
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    print("🤖 Загрузка модели Qwen2.5-VL в режиме тестирования (Inference)...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Загружаем строго на CPU в режиме оценки (inference)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"}
    )
    model.eval() # Переводим модель в режим тестирования (выключаем обучение)

    # Задаем контрольное задание
    image_path = "dummy_chart.jpg"
    user_query = "Какая аномалия на этом финтех-графике?"

    if not os.path.exists(image_path):
        print(f"❌ Файл {image_path} не найден. Пожалуйста, положите картинку в папку.")
        return

    print(f"\nТестовый запрос: '{user_query}'")
    print("Модель анализирует изображение...")

    # Подготавливаем шаблон диалога
    image = Image.open(image_path).convert("RGB")
    prompt = f"<|im_start|>user\n<|image_pad|>{user_query}<|im_end|>\n<|im_start|>assistant\n"

    # Прогоняем через процессор БЕЗ расчета градиентов (torch.no_grad)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        
    # Вырезаем входные токены, чтобы оставить только чистый ответ модели
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Декодируем токены обратно в человеческий текст
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("\n=== ОТВЕТ МОДЕЛИ НА КОНТРОЛЬНОМ ЗАДАНИИ ===")
    print(output_text[0])

if __name__ == "__main__":
    run_test_inference()