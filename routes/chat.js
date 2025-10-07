import express from "express";
import axios from "axios";
import { getDb } from "../lib/db.js";
import jwt from "jsonwebtoken";

const router = express.Router();

// auth middleware
// function auth(req, res, next) {
//   const authHeader = req.headers.authorization;
//   if (!authHeader) return res.status(401).json({ error: "unauthorized" });
//   const token = authHeader.replace("Bearer ", "");
//   try {
//     const payload = jwt.verify(token, process.env.JWT_SECRET || "changeme");
//     req.user = payload;
//     next();
//   } catch (err) {
//     return res.status(401).json({ error: "invalid token" });
//   }
// }

// helper: geocode location string -> lat/lon using Nominatim
async function geocodeLocation(query, language = "en") {
  if (!query) return null;

  const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(
    query
  )}&format=json&limit=1&accept-language=${language}`; 

  const res = await axios.get(url, {
    headers: { "User-Agent": "FarmerAdvisoryApp/1.0" },
  });

  if (res.data && res.data.length > 0) {
    return {
      lat: parseFloat(res.data[0].lat),
      lon: parseFloat(res.data[0].lon),
      displayName: res.data[0].display_name,
    };
  }
  return null;
}

// helper: fetch weather using OpenWeather API
async function fetchWeather(lat, lon, language = "en") {
  if (!lat || !lon) return null;
  const apiKey = process.env.OPENWEATHER_API_KEY;
  if (!apiKey) return null;

  const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric&lang=${language}`; // ✅ added lang
  const res = await axios.get(url);

  return {
    temperature: res.data.main.temp,
    condition: res.data.weather[0].description,
    humidity: res.data.main.humidity,
  };
}

router.post("/",  async (req, res) => {
  try {
    let {
      text,
      language = "hi", // default Hindi
      sessionId = null,
      lat = null,
      lon = null,
      state = null,
      district = null,
      pincode = null,
    } = req.body;

    if (!text) return res.status(400).json({ error: "text required" });

    // If lat/lon not provided, try to resolve via geocoding
    if (!lat || !lon) {
      const query = [pincode, district, state, "India"]
        .filter(Boolean)
        .join(", ");
      if (query) {
        const cordinates = await geocodeLocation(query, language); 
        console.log("Geocoded:", cordinates);  //debug
        if (cordinates) {
          lat = cordinates.lat;
          lon = cordinates.lon;
        }
      }
    }

    // store user message in DB
    // const db = getDb();
    // const sessions = db.collection("sessions");
    // const session = sessionId
    //   ? await sessions.findOne({ _id: sessionId })
    //   : null;
    // const sid = session
    //   ? session._id
    //   : (
    //       await sessions.insertOne({
    //         userId: req.user.sub,
    //         messages: [],
    //         createdAt: new Date(),
    //       })
    //     ).insertedId;

    // await sessions.updateOne(
    //   { _id: sid },
    //   {
    //     $push: {
    //       messages: {
    //         role: "user",
    //         text,
    //         ts: new Date(),
    //         lat,
    //         lon,
    //         language,
    //         state,
    //         district,
    //         pincode,
    //       },
    //     },
    //   }
    // );

   
    const weather = await fetchWeather(lat, lon, language); 
    console.log("Fetched weather:", weather); // debug

    // forward to AI microservice
    const aiResp = await axios.post(
      `${process.env.AI_SERVICE_URL || "http://localhost:8000"}/rag_query`,
      {
        //userId: req.user.sub,
        text,
        language,
        location: { lat, lon, state, district, pincode },
        weather,
      },
      { timeout: 45000 }
    );

    const assistantText = aiResp.data.answer || aiResp.data;
    console.log("AI Response:", assistantText); // debug

    // store assistant msg
    // await sessions.updateOne(
    //   { _id: sid },
    //   {
    //     $push: {
    //       messages: {
    //         role: "assistant",
    //         text: assistantText,
    //         ts: new Date(),
    //         meta: aiResp.data.meta || {},
    //       },
    //     },
    //   }
    // );

    res.json({
      answer: assistantText,
     // sessionId: sid,
      meta: { ...(aiResp.data.meta || {}), weather },
    });
  } catch (err) {
    console.error(err?.response?.data || err.message || err);
    res.status(500).json({ error: err.message });
  }
});

export default router;
