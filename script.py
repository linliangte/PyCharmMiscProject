from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from translator_engine import WzuNmtEngine
import uvicorn
import io
from datetime import datetime

app = FastAPI(title="英汉翻译系统")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化引擎
try:
    engine = WzuNmtEngine()
    print("✅ 多模型翻译引擎启动成功！（新闻领域 + 日常对话）")
except Exception as e:
    print(f"❌ 引擎启动失败: {e}")
    engine = None


class Msg(BaseModel):
    text: str


class SwitchModel(BaseModel):
    model_name: str


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/translate")
async def trans(m: Msg):
    if not engine:
        return {"output": "模型未加载", "status": "error"}

    try:
        output = engine.translate(m.text)
        return {
            "output": output,
            "status": "success",
            "current_model": engine.current_model,
            "model_name": engine.model_configs[engine.current_model]["name"]
        }
    except Exception as e:
        return {"output": f"翻译出错: {str(e)}", "status": "error"}


@app.post("/switch_model")
async def switch_model(req: SwitchModel):
    if not engine:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        success = engine.switch_model(req.model_name)
        if success:
            return {
                "status": "success",
                "message": f"已切换到 {engine.model_configs[req.model_name]['name']}"
            }
        else:
            raise HTTPException(status_code=400, detail="切换失败")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/get_history")
async def get_history():
    if not engine:
        raise HTTPException(status_code=500, detail="模型未加载")
    history = engine.get_history(limit=100)
    return {"status": "success", "count": len(history), "history": history}


@app.post("/clear_history")
async def clear_history():
    if not engine:
        raise HTTPException(status_code=500, detail="模型未加载")
    engine.clear_history()
    return {"status": "success", "message": "历史已清空"}


@app.post("/translate_file")
async def translate_file(
    file: UploadFile = File(...),
    model_name: str = None
):
    if not engine:
        raise HTTPException(status_code=500, detail="模型未加载")

    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="目前仅支持 .txt 文件")

    try:
        content = await file.read()
        original_text = content.decode("utf-8").strip()

        if not original_text:
            raise HTTPException(status_code=400, detail="文件内容为空")

        if model_name and model_name in engine.model_configs:
            engine.switch_model(model_name)

        translated = engine.translate(original_text)

        # 生成下载内容
        download_content = f"""=== 原文 ===\n{original_text}

=== 翻译结果 ===\n{translated}

=== 翻译信息 ===
文件名: {file.filename}
使用模型: {engine.model_configs[engine.current_model]["name"]}
翻译时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return {
            "status": "success",
            "filename": file.filename,
            "translated": translated,
            "used_model": engine.model_configs[engine.current_model]["name"],
            "download_content": download_content
        }

    except Exception as e:
        import traceback
        print("文件翻译错误:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")


# ===================== 下载接口（已修复中文文件名问题） =====================
@app.post("/download_result")
async def download_result(data: dict):
    content = data.get("content", "")
    original_filename = data.get("filename", "translated.txt")

    if not content:
        raise HTTPException(status_code=400, detail="没有可下载的内容")

    # 处理文件名：如果包含中文，使用 RFC 5987 编码方式
    if any(ord(c) > 127 for c in original_filename):
        # 对中文文件名进行安全处理
        safe_filename = "translated_result.txt"
    else:
        safe_filename = original_filename

    # 创建内存文件流
    file_obj = io.BytesIO(content.encode("utf-8"))
    file_obj.seek(0)

    return StreamingResponse(
        file_obj,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"'
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)