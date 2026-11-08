from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai_engine import SimpleMultimodalEngine

app = FastAPI(title="Легкий ИИ-Сервис на чистой Ollama")
ai_engine = SimpleMultimodalEngine()

class AnalysisRequest(BaseModel):
    pdf_path: str
    question: str
    page: int = 1 # Добавили поле номера страницы (по умолчанию 1)
@app.post("/ask_pdf")
async def ask_pdf_endpoint(request: AnalysisRequest):
    try:
        # Передаем номер страницы request.page в наш движок
        answer = ai_engine.analyze_pdf_and_answer(
            pdf_path=request.pdf_path, 
            user_question=request.question,
            target_page=request.page
        )
        return {"status": "success", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
