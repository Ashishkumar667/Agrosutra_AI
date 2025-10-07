import express from "express";
import dotenv from "dotenv";
import bodyParser from "body-parser";
import cors from "cors";
import authRouter from "./routes/auth.js";
import chatRouter from "./routes/chat.js";
import imagesRouter from "./routes/image.js";
import docsRouter from "./routes/doc.js";
import weatherRouter from "./routes/weather.js";
import marketRouter from "./routes/market.js";
import feedbackRouter from "./routes/feedback.js";
import { initDb } from "./lib/db.js";

dotenv.config();
await initDb(); // initialize Mongo connection

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: "10mb" }));

app.use("/api/v1/auth", authRouter);
app.use("/api/v1/chat", chatRouter);
app.use("/api/v1/images", imagesRouter);
app.use("/api/v1/docs", docsRouter);
app.use("/api/v1/weather", weatherRouter);
app.use("/api/v1/market", marketRouter);
app.use("/api/v1/feedback", feedbackRouter);

// health
app.get("/health", (req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Node API running on ${PORT}`));
