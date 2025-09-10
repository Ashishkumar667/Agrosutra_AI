import express from "express";
import axios from "axios";
const router = express.Router();

// Example: Fetch mandi price from Agmarknet / eNAM (public APIs differ per provider).
router.get("/price", async (req, res) => {
  try {
    const { commodity = "Tomato", state = "Delhi", date } = req.query;
    // This is a stub: you should replace with real API calls and parsing.
    // For demo, we return a mocked response or attempt to fetch Agmarknet if available.
    const sample = { commodity, state, mandi: "Delhi Mandi", price: Math.floor(10 + Math.random() * 50), unit: "₹/kg", date: date || new Date().toISOString().slice(0,10) };
    res.json(sample);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
