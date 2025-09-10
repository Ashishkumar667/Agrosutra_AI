import express from "express";
import axios from "axios";
import { getDb } from "../lib/db.js";
import jwt from "jsonwebtoken";

const router = express.Router();

function auth(req,res,next){
  const authHeader = req.headers.authorization;
  if (!authHeader) return res.status(401).json({error:"unauthorized"});
  const token = authHeader.replace("Bearer ","");
  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET || "changeme");
    req.user = payload;
    next();
  } catch(err){ return res.status(401).json({error:"invalid token"}); }
}

router.post("/", auth, async (req, res) => {
  try {
    const { text, language = "hi", sessionId = null, lat = null, lon = null } = req.body;
    if (!text) return res.status(400).json({ error: "text required" });

    // store user message in DB
    const db = getDb();
    const sessions = db.collection("sessions");
    const session = sessionId ? await sessions.findOne({ _id: sessionId }) : null;
    const sid = session ? session._id : (await sessions.insertOne({ userId: req.user.sub, messages: [], createdAt: new Date() })).insertedId;

    await sessions.updateOne({ _id: sid }, { $push: { messages: { role: "user", text, ts: new Date(), lat, lon, language } } });

    // forward to AI microservice
    const aiResp = await axios.post(`${process.env.AI_SERVICE_URL || "http://localhost:8000"}/rag_query`, {
      userId: req.user.sub,
      text,
      language,
      location: { lat, lon }
    }, { timeout: 45000 });

    const assistantText = aiResp.data.answer || aiResp.data;

    // store assistant msg
    await sessions.updateOne({ _id: sid }, { $push: { messages: { role: "assistant", text: assistantText, ts: new Date(), meta: aiResp.data.meta || {} } } });

    res.json({ answer: assistantText, sessionId: sid, meta: aiResp.data.meta || {} });
  } catch (err) {
    console.error(err?.response?.data || err.message || err);
    res.status(500).json({ error: err.message });
  }
});

export default router;
