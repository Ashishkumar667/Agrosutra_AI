import express from "express";
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
    const { type, itemId, rating = null, comment = null } = req.body;
    const db = getDb();
    const coll = db.collection("feedback");
    await coll.insertOne({ userId: req.user.sub, type, itemId, rating, comment, createdAt: new Date() });
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
