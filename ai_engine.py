import os
from pdf2image import convert_from_path

class SimpleMultimodalEngine:
    def __init__(self):
        # Ваши две локальные модели Ollama
        self.vision_model = "qwen2.5vl"
        self.text_model = "qwen2.5:1.5b"
        
        # Папка для временного хранения картинок страниц
        self.temp_folder = "./temp_pages"
        os.makedirs(self.temp_folder, exist_ok=True)
        
    def analyze_pdf_and_answer(self, pdf_path: str, user_question: str, target_page: int) -> str:
        """Сверхбыстрый конвейер: прыгает сразу на нужную страницу"""
        if not os.path.exists(pdf_path):
            return f"Ошибка: Файл {pdf_path} не найден."
            
        print(f" Делаем прыжок! Загружаем строго страницу {target_page}...")
        
        # Нарезаем из PDF ТОЛЬКО одну конкретную страницу для экономии времени
        pages = convert_from_path(
            pdf_path, 
            dpi=120, 
            first_page=target_page,
            last_page=target_page,
            poppler_path=r"E:\BI_разработчик\Ollama\MultiModal_LLM\Release-26.02.0-0\poppler-26.02.0\Library\bin"
        )
        
        if not pages:
            return f"Ошибка: Не удалось извлечь страницу {target_page}."
            
        import ollama  # Локальная библиотека Ollama
        extracted_context = []
        
        # Извлекаем единственную страницу из списка
        page = pages[0]
        image_path = os.path.join(self.temp_folder, f"target_page_{target_page}.jpg")
        page.save(image_path, "JPEG")  # Сохраняем как .jpg
        
        # ШАГ 1: Зрячая модель анализирует картинку и переводит её в текст
        print(f" Зрячая модель {self.vision_model} анализирует страницу {target_page}...")
        
        #response = ollama.chat(
        #   model=self.vision_model,
        #   messages=[{
        #       'role': 'user',
        #       'content': 'Внимательно изучи эту страницу. Опиши текстом всё, что ты видишь: текст, таблицы, значения на осях графиков и тренды.',
        #       'images': [image_path]
        #   }]
        #)
        response = ollama.chat(
            model=self.vision_model,
            messages=[{
                'role': 'user',
                'content': (
                    "Ты — главный системный аналитик и эксперт по чтению сложных UML/архитектурных диаграмм. "
                    "Перед тобой сложнейшая диаграмма трассировки классов анализа в классы проектирования. "
                    "Действуй по шагам (Chain-of-Thought):\n"
                    "1. Перечисли ВСЕ ключевые классы и таблицы, которые ты видишь на схеме (например, Пользователь, Статья, Article, Resource, VideoMapper и т.д.).\n"
                    "2. Для каждого найденного класса выпиши несколько его внутренних полей, методов или типов данных, которые удается четко разобрать (например, ISBN: String, createTable(), id: long).\n"
                    "3. Опиши структуру связей: какие блоки соединены стрелками с пометками «trace» или линиями отношений.\n"
                    "Выдай максимально подробный, детализированный технический разбор. Не обобщай, пиши факты!"
                ),
                'images': [image_path]
            }]
        )
        
        extracted_context.append(response['message']['content'])
        
        # Удаляем временную картинку, чтобы не забивать диск
        if os.path.exists(image_path):
            os.remove(image_path)
                
        full_document_text = "\n\n".join(extracted_context)
        
        # ШАГ 2: Текстовая модель объединяет описание графиков и текстовый вопрос
        print(f" Текстовая модель {self.text_model} формирует финальный финтех-ответ...")
        
        final_prompt = (
            f"Используй только следующее описание документа для ответа на вопрос.\n"
            f"Описание страницы {target_page}: {full_document_text}\n"
            f"Вопрос пользователя: {user_question}\n"
            f"Ответ:"
        )
        
        final_response = ollama.chat(
            model=self.text_model,
            messages=[{'role': 'user', 'content': final_prompt}]
        )
        
        return final_response['message']['content']

