from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import uvicorn
import os, io, json
import time
from dotenv import load_dotenv
import requests
import numpy as np
import glob
import PyPDF2

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/pest_model.pth")

app = FastAPI(title="AI Microservice for Agri Chatbot")


# Try to lazy-load a PyTorch model if available
MODEL = None
CLASSES = None
try:
    import torch
    from torchvision import transforms, models
    from PIL import Image
    print("Torch available. Attempting to load model...")
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        CLASSES = ckpt.get("classes", None)
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        MODEL = model
        print("Loaded pest model from", MODEL_PATH)
    else:
        print("No model found at", MODEL_PATH, "- CV endpoints will return fallback predictions.")
except Exception as e:
    print("Torch unavailable or model load failed:", str(e))
    MODEL = None

# SOIL_TEXTS = {}

def load_soil_pdfs(folder="data/soil_reports"):
    texts = {}
    for pdf_file in glob.glob(os.path.join(folder, "*.pdf")):
        try:
            with open(pdf_file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([p.extract_text() or "" for p in reader.pages])
                state_name = os.path.splitext(os.path.basename(pdf_file))[0].lower()
                texts[state_name] = text
                print(f"Loaded soil PDF: {state_name}")
        except Exception as e:
            print("Failed loading", pdf_file, str(e))
    return texts

# ============= Soil RAG Setup ==============
INDEX_DIR = "soil_index"
PDF_FOLDER = "data/soil_reports"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

soil_index = None
pdf_timestamps = {}  # track modification times

def build_soil_index(folder=PDF_FOLDER):
    docs = []
    for pdf_file in glob.glob(os.path.join(folder, "*.pdf")):
        try:
            with open(pdf_file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([p.extract_text() or "" for p in reader.pages])
                state_name = os.path.splitext(os.path.basename(pdf_file))[0].lower()
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_text(text)
                for c in chunks:
                    docs.append({"state": state_name, "text": c})
                print(f"Indexed {len(chunks)} chunks for {state_name}")
        except Exception as e:
            print("Failed loading", pdf_file, str(e))
    if not docs:
        return None
    texts = [d["text"] for d in docs]
    metadatas = [{"state": d["state"]} for d in docs]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore

# Add this test endpoint to debug
@app.get("/debug_soil_index")
async def debug_soil_index():
    """Debug endpoint to check soil index status"""
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    return {
        "pdf_files": pdf_files,
        "index_exists": os.path.exists(INDEX_DIR),
        "soil_index_loaded": soil_index is not None
    }
    
def load_or_build_index():
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return build_soil_index()

def watch_soil_reports(interval=30):
    global soil_index, pdf_timestamps
    while True:
        changed = False
        for pdf_file in glob.glob(os.path.join(PDF_FOLDER, "*.pdf")):
            mtime = os.path.getmtime(pdf_file)
            if pdf_file not in pdf_timestamps or pdf_timestamps[pdf_file] < mtime:
                print(f"[Watcher] Detected new/updated PDF: {pdf_file}")
                changed = True
                pdf_timestamps[pdf_file] = mtime
        if changed:
            soil_index = build_soil_index()
            print("[Watcher] Soil index refreshed")
        time.sleep(interval)



# =================== OpenAI helper ====================
def call_openai_answer(prompt: str, max_tokens=600):
    if not OPENAI_KEY:
        return f"[LLM fallback answer] {prompt[:200]}"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role":"system","content":"You are an expert agricultural assistant."},
                     {"role":"user","content":prompt}],
        "max_tokens": max_tokens
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# =================== Soil Info Endpoint ====================
class SoilInfoRequest(BaseModel):
    state: str
    district: str | None = None
    language: str | None = "hi"

@app.post("/soil_info")
async def soil_info(r: SoilInfoRequest):
    """
    Summarize soil health for a given state/district using FAISS index.
    """
    global soil_index
    if soil_index is None:
        return JSONResponse(
            {"error": "Soil index not available. Please add soil PDFs and wait for refresh."},
            status_code=404
        )

    # Query index
    query = f"Soil health for {r.state} {r.district or ''}"
    print("Soil info query:", query)
    docs = soil_index.similarity_search(query, k=3)
    if not docs:
        return JSONResponse({"error": "No relevant soil data found."}, status_code=404)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
आप एक कृषि विशेषज्ञ सहायक हैं। नीचे दिए गए {r.state} राज्य {r.district or ''} जिले की मृदा स्वास्थ्य रिपोर्ट के आधार पर किसान को सलाह दें।

रिपोर्ट (संदर्भ):
{context}

कार्य:
1. मृदा स्वास्थ्य की स्थिति का संक्षिप्त वर्णन करें।
2. 3 उपयुक्त फसलों की अनुशंसा करें।
3. प्रत्येक फसल के लिए उर्वरक (खाद) के नाम {r.language} में बताएं।
4. उत्तर केवल {r.language} भाषा में दें। 

उत्तर को किसान के समझने योग्य सरल शब्दों में लिखें।
    """

    answer = call_openai_answer(prompt)
    print("Soil info answer:", answer)
    print("prompt:", prompt)
    return JSONResponse({"answer": answer, "source": r.state})



@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...), userId: str = Form(None)):
    """
    Receives an image file, runs pest/disease detection (if model available) 
    and returns: { label, confidence, severity, advice }
    """
    content = await file.read()

    # --- Temporary testing fallback ---
    if MODEL is None:
        fake_label = "leaf_blight"
        fake_conf = 0.85
        fake_severity = "moderate"
        fake_advice = "प्रभावित पत्तों को हटा दें, कॉपर आधारित फफूंदनाशी का छिड़काव करें, और फसल चक्र अपनाएँ।"

        return JSONResponse({
            "label": fake_label,
            "confidence": fake_conf,
            "severity": fake_severity,
            "advice": fake_advice,
            "meta": {"note": "⚠️ Using fake prediction because model is not loaded"}
        })

    # --- Real model inference if MODEL is loaded ---
    try:
        from PIL import Image
        import torch
        from torchvision import transforms
        image = Image.open(io.BytesIO(content)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            out = MODEL(x)
            probs = torch.softmax(out, dim=1).numpy()[0]
            idx = int(np.argmax(probs))
            label = CLASSES[idx] if CLASSES else str(idx)
            conf = float(probs[idx])

            severity = "mild" if conf > 0.6 else "moderate" if conf > 0.4 else "low"

            # LLM advice
            prompt = f"किसान की फसल में '{label}' बीमारी पाई गई है (विश्वास स्तर {conf:.2f}). छोटे किसान के लिए 3 उपाय बताइए (तत्काल, मध्यम अवधि, लंबी अवधि)। सरल हिंदी में।"
            advice = call_openai_answer(prompt)

            return JSONResponse({
                "label": label,
                "confidence": conf,
                "severity": severity,
                "advice": advice
            })
    except Exception as e:
        return JSONResponse({"label": "error", "error": str(e)}, status_code=500)


class RAGRequest(BaseModel):
    userId: str | None = None
    text: str
    language: str | None = "hi"
    location: dict | None = None

@app.post("/rag_query")
async def rag_query(req: RAGRequest):
    """
    Basic RAG flow implemented simply:
    - For hackathon starter: call OpenAI chat completion with a RAG-style prompt.
    - In full implementation: you'd query vector DB and include retrieved chunks.
    """
    text = req.text
    # In a real implementation you would:
    # 1) transform query to embedding
    # 2) query vector DB (Pinecone/Weaviate) for top-k chunks
    # 3) construct prompt with context and call LLM
    # For hackathon starter, we call LLM directly with instruction to be concise and cite sources if available.
    prompt = f"""
You are an agricultural assistant for small farmers in India. The user asks: {text}
Provide a 1-2 sentence summary, then three action steps (immediate, next 3 days, prevention). Use simple words in {req.language}. If you need weather/soil data say what you need.
"""
    answer = call_openai_answer(prompt)
    return JSONResponse({ "answer": answer, "meta": { "provider": "openai-fallback" } })

class SoilRequest(BaseModel):
    ph: float | None = None
    n: float | None = None
    p: float | None = None
    k: float | None = None
    oc: float | None = None
    crop: str | None = None
    state: str | None = None

@app.post("/soil_recommendation")
async def soil_recommendation(r: SoilRequest):
    """
    Basic rules-based fertilizer guidance. In production replace with state-specific mapping.
    """
    # Default safe guidance
    ph = r.ph
    n = r.n
    p = r.p
    k = r.k
    crop = r.crop or "your crop"
    # Simple rule examples:
    advice = []
    if ph:
        if ph < 6.0:
            advice.append("Soil is acidic. Apply lime as per local recommendations.")
        elif ph > 7.8:
            advice.append("Soil is alkaline. Consider gypsum and organic matter.")
        else:
            advice.append("Soil pH is OK.")
    if n is not None and p is not None and k is not None:
        # naive NPK suggestion example
        # In production, map to state guidelines
        advice.append(f"For {crop}, consider N:{max(0,50-int(n))} kg/ha, P:{max(0,30-int(p))} kg/ha, K:{max(0,20-int(k))} kg/ha as a starting point. Confirm with local KVK.")
    # fallback to LLM for explanation
    prompt = f"Given these soil readings pH={ph}, N={n}, P={p}, K={k}, give a short fertilizer plan in simple language for {crop}."
    explanation = call_openai_answer(prompt)
    return JSONResponse({ "advice_rules": advice, "explanation": explanation })

# Simple STT endpoint using OpenAI Whisper via the OpenAI API if key is available
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    audio = await file.read()
    if not OPENAI_KEY:
        return JSONResponse({"text": "[STT fallback] (no OPENAI_KEY configured)"})
    # Send to OpenAI speech-to-text endpoint (example)
    import requests, json
    files = {"file": ("audio.wav", audio)}
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    # Note: This is a conceptual example - use official SDK or specified endpoint for your provider
    resp = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files, timeout=60)
    if resp.status_code != 200:
        return JSONResponse({"error": resp.text}, status_code=500)
    data = resp.json()
    return JSONResponse({"text": data.get("text")})

# TTS (text -> audio) using an external TTS provider (stub)
@app.post("/tts")
async def tts(text: str = Form(...), voice: str = Form("default")):
    # For demo return a small JSON with URL to generated audio in production
    if not OPENAI_KEY:
        return JSONResponse({"audio_url": None, "note": "TTS provider not configured. Use ElevenLabs or Google Cloud TTS."})
    # Implement provider calls here. For hackathon return a placeholder.
    return JSONResponse({"audio_url": None, "note": "TTS not configured in this demo."})


# if __name__ == "__main__":
#     # Load index once
#     soil_index = load_or_build_index()
#     print("Soil index initialized:", soil_index)

#     # Start background watcher thread
#     import threading
#     t = threading.Thread(target=watch_soil_reports, args=(30,), daemon=True)
#     t.start()

#     uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AI_SERVICE_PORT", 8000)))
@app.on_event("startup")
async def init_soil_index():
    global soil_index
    soil_index = load_or_build_index()
    print("🌱 Soil index initialized:", soil_index is not None)

    # Start watcher thread
    import threading
    t = threading.Thread(target=watch_soil_reports, args=(30,), daemon=True)
    t.start()
    print("👀 Soil watcher started")

