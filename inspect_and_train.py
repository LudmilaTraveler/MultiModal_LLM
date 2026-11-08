import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image

# =====================================================================
# БЛОК 1: ИНСПЕКЦИЯ СЛОЕВ И ИХ ВЕСОВ
# =====================================================================
def inspect_and_print_layers(model):
    print("\n=== АНАЛИЗ СТРУКТУРЫ СЛОЕВ И ИХ СОСТОЯНИЯ ===")
    trainable_count = 0
    frozen_count = 0
    
    # Проходим по всем весам модели построчно
    for name, param in model.named_parameters():
        # param.numel() — считает количество числовых весов (параметров) в слое
        if param.requires_grad:
            trainable_count += param.numel()
            # Выводим веса, которые ОСТАЛИСЬ АКТИВНЫМИ для обучения (наши адаптеры)
            print(f"🔥 ОБУЧАЕМЫЙ СЛОЙ: {name} | Формат: {param.dtype} | Весов: {param.numel()}")
        else:
            frozen_count += param.numel()
            
    print(f"\nЗаморожено базовых весов: {frozen_count:,}")
    print(f"Активно для обучения (LoRA): {trainable_count:,}")
    print(f"Доля обучаемых параметров: {100 * trainable_count / (frozen_count + trainable_count):.4f}%\n")


# =====================================================================
# БЛОК 2: КЛАСС ПОДГОТОВКИ ДАТАСЕТА (PyTorch Dataset)
# =====================================================================
class FintechTrainDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        self.processor = processor
        
        # Читаем JSONL файл построчно (наш датасет)
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            print(f"⚠️ Файл датасета {jsonl_path} не найден! Будет создан тестовый пример.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Загружаем картинку (график или чек), указанную в строке JSONL
        image = Image.open(item["image"]).convert("RGB")
        
        # 2. Вытаскиваем промпт пользователя и эталонный ответ ассистента
        user_query = item["messages"][0]["content"]
        assistant_reply = item["messages"][1]["content"]

        # 3. Формируем специальный шаблон диалога (Prompt Template) под Qwen2.5-VL
        prompt = f"<|im_start|>user\n<|image_pad|>{user_query}<|im_end|>\n<|im_start|>assistant\n{assistant_reply}<|im_end|>"

        # 4. Процессор превращает текст в токены (числа), а картинку — в пиксельные тензоры
        inputs = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        
        # Убираем лишнюю размерность батча, которую процессор создает по умолчанию
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Копируем входные токены в графу "labels" (ответы), чтобы модель знала, с чем себя сверять
        inputs["labels"] = inputs["input_ids"].clone() 
        return inputs


# =====================================================================
# БЛОК 3: ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ И НАСТРОЙКА АДАПТЕРОВ (LoRA)
# =====================================================================
def start_fintech_training():
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # 1. Загружаем процессор и базовую модель в bfloat16 для жесткой экономии ОЗУ
    print("Шаг 1: Загрузка базовой архитектуры Qwen2.5-VL...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"}  # Направляем всю модель строго на процессор (CPU)       
    )

    # 2. ЖЕСТКАЯ ЗАМОРОЗКА: Выключаем расчет градиентов для ВСЕЙ базовой модели
    model.requires_grad_(False)

    # 3. НАСТРОЙКА LoRA: Задаем конфигурацию наших легких адаптеров
    print("Шаг 2: Наложение низкоранговых LoRA-адаптеров...")
    lora_config = LoraConfig(
        r=16,             # Ранг адаптера (размер матрицы весов). 16 — оптимум для финтеха
        lora_alpha=32,    # Коэффициент масштабирования весов адаптера
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Обучаем только слои внимания текста
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Обертываем замороженную модель в адаптеры
    model = get_peft_model(model, lora_config)

    # 4. ВЫЗОВ ИНСПЕКТОРА СЛОЕВ (Блок 1)
    inspect_and_print_layers(model)

    # 5. ПОДГОТОВКА СИНТЕТИЧЕСКИХ ДАТАСЕТОВ (Если файлы отсутствуют)
    # Нам нужен реальный график, используем ваш dummy_chart.jpg из папки (сделайте его >0 КБ!)
    if not os.path.exists("train_dataset.jsonl"):
        sample_data = {
            "image": "dummy_chart.jpg",
            "messages": [
                {"role": "user", "content": "Какая аномалия на графике?"},
                {"role": "assistant", "content": "В третьем квартале зафиксировано аномальное падение выручки до нуля."}
            ]
        }
        with open("train_dataset.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
        print("[Инфо]: Создан тестовый train_dataset.jsonl для проверки.")

    # 6. ИНИЦИАЛИЗАЦИЯ ЗАГРУЗЧИКА ДАННЫХ (DataLoader)
    train_dataset = FintechTrainDataset("train_dataset.jsonl", processor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # batch_size=1 для 8ГБ памяти!

    # 7. ОПТИМИЗАТОР И ЗАПУСК ОБУЧЕНИЯ (Один контрольный шаг)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    print("Шаг 3: Запуск контрольного шага обучения нейросети...")
    for batch in train_loader:
        # Переносим батч на видеокарту (если доступна CUDA)
        device = "cpu"
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Прямой проход (Модель смотрит на график и пытается угадать ответ)
        outputs = model(**batch)
        loss = outputs.loss
        
        # Обратный проход (Градиентный спуск — вычисляем ошибку и обновляем веса ЛОРА)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"🔥 [УСПЕХ]: Шаг обучения выполнен! Текущая ошибка (Loss): {loss.item():.4f}")
        break # Останавливаем после 1 шага для контроля системы

    # =====================================================================
    # БЛОК 4: ОСОБОЕ МЕСТО ХРАНЕНИЯ ВЕСОВ АДАПТЕРОВ
    # =====================================================================
    # Метод save_pretrained сохраняет исключительно маленькие веса адаптеров!
    output_dir = "./lora_weights"
    model.save_pretrained(output_dir)
    print(f"Шаг 4: Веса LoRA-адаптеров успешно сохранены отдельно в папку: '{output_dir}'")

if __name__ == "__main__":
    start_fintech_training()
