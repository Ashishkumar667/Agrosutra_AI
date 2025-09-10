import express from "express";
import multer from "multer";
import { getDb } from "../lib/db.js";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import axios from "axios";
import FormData from "form-data";

const upload = multer({ dest: "tmp/docs/" });
const router = express.Router();

router.post("/upload", upload.single("file"), async (req, res) => {
  try {
    // Save file metadata to DB and enqueue for ingestion by Python AI service or worker
    const db = getDb();
    const docs = db.collection("documents");
    const id = uuidv4();
    const saved = await docs.insertOne({
      docId: id,
      filename: req.file.originalname,
      path: req.file.path,
      status: "uploaded",
      uploadedAt: new Date()
    });

    // forward to AI service ingest endpoint (optional)
    const form = new FormData();
    form.append("file", fs.createReadStream(req.file.path));
    form.append("docId", id);
    const aiUrl = `${process.env.AI_SERVICE_URL || "http://localhost:8000"}/ingest_doc`;
    try {
      await axios.post(aiUrl, form, { headers: form.getHeaders(), timeout: 120000 });
      // AI service will handle ingestion and updating vector DB
    } catch(e){
      console.warn("AI ingest failed (will retry):", e.message);
    }

    res.json({ docId: id, status: "uploaded" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

export default router;
