import requests

SERVER_URL = "http://localhost:8000/ask_pdf"

def test_multimodal_system():
    print("--- Тестирование системы Сестра-1 ---")
    
    pdf_path = r"E:\BI_разработчик\Ollama\MultiModal_LLM\My_articles\ИУ5-45МВ_Литвинович-РПЗ1.pdf"
    question = "Что изображено в диаграмме на этой странице?"
    
    # Теперь мы передаем конкретную страницу прямо в запросе!
    payload = {
        "pdf_path": pdf_path,
        "question": question,
        "page": 34 
    }
    
    print(f"\nОтправка запроса на страницу {payload['page']}... (Анализ займет около минуты)")
    try:
        response = requests.post(SERVER_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== ОТВЕТ СИСТЕМЫ ===")
            print(result.get("answer"))
        else:
            print(f"Ошибка сервера: {response.status_code}, {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Ошибка: Сервер FastAPI не запущен! Сначала запустите main.py")

if __name__ == "__main__":
    test_multimodal_system()
