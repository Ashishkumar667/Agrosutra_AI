import express from "express";
import multer from "multer";
import fs from "fs";
import FormData from "form-data";
import axios from "axios";
import jwt from "jsonwebtoken";
import { getDb } from "../lib/db.js";

const upload = multer({ dest: "tmp/uploads/" });
const router = express.Router();

// function auth(req,res,next){
//   const authHeader = req.headers.authorization;
//   if (!authHeader) return res.status(401).json({error:"unauthorized"});
//   const token = authHeader.replace("Bearer ","");
//   try {
//     const payload = jwt.verify(token, process.env.JWT_SECRET || "changeme");
//     req.user = payload;
//     next();
//   } catch(err){ return res.status(401).json({error:"invalid token"}); }
// }

router.post("/upload", upload.single("image"), async (req, res) => {
  try {
    const localPath = req.file.path;
    console.log("req file", localPath);
    // forward to AI service /predict_image
    const form = new FormData();
    form.append("file", fs.createReadStream(localPath));
    //form.append("userId", req.user.sub);
    const aiUrl = `${process.env.AI_SERVICE_URL || "http://localhost:8000"}/predict_image`;
    const aiResp = await axios.post(aiUrl, form, { headers: { ...form.getHeaders() }, timeout: 60000 });
console.log("AI Response:", aiResp.data); // debug
    // store ImageReport in DB
    const db = getDb();
    const reports = db.collection("imageReports");
    const report = {
      //userId: req.user.sub,
      createdAt: new Date(),
      fileMeta: { originalname: req.file.originalname, mimetype: req.file.mimetype },
      aiResult: aiResp.data
    };
    await reports.insertOne(report);

    // cleanup local file
    fs.unlinkSync(localPath);
    res.json({ report: aiResp.data });
  } catch (err) {
    console.error(err?.response?.data || err.message || err);
    res.status(500).json({ error: err.message });
  }
});

export default router;
