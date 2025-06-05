from fastapi import FastAPI, WebSocket
import whisper
import openai
import base64
import json
import tempfile
import os

app = FastAPI()
model = whisper.load_model("base")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = b""

    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if message["event"] == "media":
                chunk = base64.b64decode(message["media"]["payload"])
                audio_data += chunk

            elif message["event"] == "stop":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_data)
                    audio_path = f.name

                result = model.transcribe(audio_path)
                user_text = result["text"]

                openai.api_key = os.getenv("OPENAI_API_KEY")
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": user_text}],
                )
                bot_reply = response["choices"][0]["message"]["content"]
                await websocket.send_text(json.dumps({"event": "bot_reply", "text": bot_reply}))
                os.remove(audio_path)
                break

    except Exception as e:
        print("❌ Error:", e)
        await websocket.close()
