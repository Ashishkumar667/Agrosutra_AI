import express from "express";
import axios from "axios";
import dotenv from "dotenv";
const router = express.Router();

// Fetch mandi price from Agmarknet Open Data API
router.get("/price", async (req, res) => {
  try {
    const { commodity = "Tomato", state = "Delhi", date } = req.query;
    const apiKey = process.env.AGMARKNET_API; // Replace with your Data.gov.in API key
    const resourceId = "9ef84268-d588-465a-a308-a864a43d0070"; // Agmarknet resource ID

    // Build API URL
    let apiUrl = `https://api.data.gov.in/resource/${resourceId}?api-key=${apiKey}&format=json&filters[commodity]=${encodeURIComponent(commodity)}&filters[state]=${encodeURIComponent(state)}`;
    if (date) {
      apiUrl += `&filters[arrival_date]=${encodeURIComponent(date)}`;
    }

    // Fetch data from Agmarknet
    const response = await axios.get(apiUrl);
    const records = response.data.records;

    if (!records || records.length === 0) {
      return res.status(404).json({ error: "No mandi price found for given parameters." });
    }

    // Pick the first record (or process as needed)
    const mandiData = records[0];
    const result = {
      commodity: mandiData.commodity,
      state: mandiData.state,
      mandi: mandiData.market,
      price: mandiData.modal_price,
      unit: "₹/quintal",
      date: mandiData.arrival_date
    };

      const aiResp = await axios.post(
      `${process.env.AI_SERVICE_URL || "http://localhost:8000"}/market_price_analysis`,
      {
        mandiData,
        result,
        records,
        response
      },
      { timeout: 45000 }
    );

    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;