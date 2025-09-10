import express from "express";
import { getDb } from "../lib/db.js";
import { v4 as uuidv4 } from "uuid";
import jwt from "jsonwebtoken";

const router = express.Router();

// Send OTP (stub)
router.post("/send-otp", async (req, res) => {
  const { phone, name, language = "hi" } = req.body;
  if (!phone) return res.status(400).json({ error: "phone required" });
  const db = getDb();
  const users = db.collection("users");
  let user = await users.findOne({ phone });
  if (!user) {
    user = { phone, name: name || null, language, createdAt: new Date() };
    await users.insertOne(user);
  }
  // In real prod: send OTP and save OTP hash. Here: return OTP token stub.
  const otpToken = uuidv4().split("-")[0]; // fake OTP token
  // return token in response (for dev); in production don't do this
  res.json({ message: "otp_sent (dev)", otpToken, userId: user._id });
});

// Verify OTP (stub) -> issue JWT
router.post("/verify-otp", async (req, res) => {
  const { phone, otpToken } = req.body;
  if (!phone || !otpToken) return res.status(400).json({ error: "phone & otpToken required" });
  // In real prod, verify OTP
  const db = getDb();
  const users = db.collection("users");
  const user = await users.findOne({ phone });
  if (!user) return res.status(404).json({ error: "user not found" });
  const token = jwt.sign({ sub: String(user._id), phone }, process.env.JWT_SECRET || "changeme", { expiresIn: "30d" });
  res.json({ token, user: { id: user._id, phone: user.phone, language: user.language } });
});

export default router;
