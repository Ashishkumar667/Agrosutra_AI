import express from "express";
import axios from "axios";
const router = express.Router();

// Example weather fetch using OpenWeatherMap (replace with IMD API if you have access)
router.get("/current", async (req, res) => {
  try {
    const { lat, lon } = req.query;
    if (!lat || !lon) return res.status(400).json({ error: "lat & lon required" });
    // best to cache on node side
    const apiKey = process.env.OPENWEATHER_API_KEY || "";
    if (!apiKey) return res.status(500).json({ error: "OPENWEATHER_API_KEY not configured" });
    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`;
    const resp = await axios.get(url);
    res.json(resp.data);
  } catch (err) {
    console.error(err?.response?.data || err.message);
    res.status(500).json({ error: err.message });
  }
});

export default router;
