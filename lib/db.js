import { MongoClient } from "mongodb";
import dotenv from "dotenv";
dotenv.config();

const uri = process.env.MONGO_URI || "mongodb://localhost:27017/agri_chatbot";
const client = new MongoClient(uri, {});

let db;
export async function initDb() {
  await client.connect();
  db = client.db();
  // create indexes if needed
  await db.collection("users").createIndex({ phone: 1 }, { unique: true });
  console.log("Connected to MongoDB");
}
export function getDb() {
  if (!db) throw new Error("DB not initialized");
  return db;
}
