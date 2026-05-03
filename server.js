import express from "express";
import cors from "cors";
import fs from "fs/promises";
import { createWriteStream } from "fs";
import { pipeline } from "stream/promises";
import path from "path";
import os from "os";

import { S3Client, ListObjectsV2Command } from "@aws-sdk/client-s3";
import { Client as NotionClient } from "@notionhq/client";
import OpenAI from "openai";
import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";

const app = express();

// STEP 1) allow browser button calls + parse JSON
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// STEP 2) ENV
const {
  // R2
  R2_ACCOUNT_ID,
  R2_ACCESS_KEY_ID,
  R2_SECRET_ACCESS_KEY,
  R2_BUCKET_NAME,
  R2_PUBLIC_BASE_URL,

  // Notion
  NOTION_API_KEY,
  NOTION_PAGE_ID,

  // OpenAI
  OPENAI_API_KEY,
  OPENAI_VISION_MODEL,

  // Airtable ✅ NEW
  AIRTABLE_API_KEY,
  AIRTABLE_BASE_ID,
  AIRTABLE_TABLE_NAME,
} = process.env;

// STEP 3) clients
const r2 = new S3Client({
  region: "auto",
  endpoint: `https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: R2_ACCESS_KEY_ID,
    secretAccessKey: R2_SECRET_ACCESS_KEY,
  },
});

const notion = new NotionClient({ auth: NOTION_API_KEY });
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

ffmpeg.setFfmpegPath(ffmpegPath);

// STEP 4) helper: list ALL object KEYS (paginated)
async function listAllUploadKeys() {
  const keys = [];
  let ContinuationToken = undefined;

  while (true) {
    const command = new ListObjectsV2Command({
      Bucket: R2_BUCKET_NAME,
      MaxKeys: 1000,
      ContinuationToken,
      // If you already upload into a folder, optionally add:
      // Prefix: "your-folder/",
    });

    const response = await r2.send(command);
    const files = response.Contents || [];

    for (const f of files) {
      if (f?.Key) keys.push(f.Key);
    }

    if (!response.IsTruncated) break;
    ContinuationToken = response.NextContinuationToken;
  }

  return keys;
}

// STEP 5) helper: read Notion property "Prompt"
async function getNotionPrompt() {
  if (!NOTION_API_KEY) throw new Error("Missing env var: NOTION_API_KEY");
  if (!NOTION_PAGE_ID) throw new Error("Missing env var: NOTION_PAGE_ID");

  const page = await notion.pages.retrieve({ page_id: NOTION_PAGE_ID });
  const prop = page?.properties?.Prompt;

  const textParts =
    prop?.type === "title"
      ? prop.title
      : prop?.type === "rich_text"
      ? prop.rich_text
      : null;

  if (!textParts) {
    return `[Prompt property is type "${prop?.type ?? "unknown"}" — not title/rich_text]`;
  }

  const promptText = textParts.map((t) => t.plain_text).join("").trim();
  return promptText || "[Prompt is empty]";
}

// STEP 6) helper: build public video URL from key
function buildPublicVideoUrl(key) {
  if (!R2_PUBLIC_BASE_URL) throw new Error("Missing env var: R2_PUBLIC_BASE_URL");
  const base = R2_PUBLIC_BASE_URL.replace(/\/+$/, "");
  const safeKey = key
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
  return `${base}/${safeKey}`;
}

// STEP 7) helper: download video to /tmp (STREAMING, low memory)
async function downloadToTempFile(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download video (${res.status})`);
  if (!res.body) throw new Error("No response body to stream");

  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "video-"));
  const videoPath = path.join(tmpDir, "input.mp4");

  // Stream MP4 directly to disk (avoids loading entire file into RAM)
  await pipeline(res.body, createWriteStream(videoPath));

  return { tmpDir, videoPath };
}

// STEP 8) helper: extract frames (base64 images) using ffmpeg (SMALLER + FEWER)
async function extractFramesBase64(videoPath, { fps = 1, maxFrames = 4 } = {}) {
  const framesDir = path.join(path.dirname(videoPath), "frames");
  await fs.mkdir(framesDir, { recursive: true });

  await new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      // scale down frames + lower JPEG quality = much smaller memory/payload
      .outputOptions([`-vf fps=${fps},scale=320:-1`, "-q:v 8"])
      .output(path.join(framesDir, "frame-%03d.jpg"))
      .on("end", resolve)
      .on("error", reject)
      .run();
  });

  const files = (await fs.readdir(framesDir))
    .filter((f) => f.toLowerCase().endsWith(".jpg"))
    .sort()
    .slice(0, maxFrames);

  const images = [];
  for (const f of files) {
    const p = path.join(framesDir, f);
    const b = await fs.readFile(p);
    images.push(b.toString("base64"));
  }
  return images;
}

// STEP 9) helper: OpenAI vision analysis (prompt + frames)
// NOTE: keep returning "text" as you do now
async function analyzeVideoWithOpenAI({ promptText, framesBase64, videoUrl }) {
  if (!OPENAI_API_KEY) throw new Error("Missing env var: OPENAI_API_KEY");
  const model = OPENAI_VISION_MODEL || "gpt-4.1-mini";

  const content = [
    {
      type: "input_text",
      text:
        `Analyze this stock video for retrieval.\n` +
        `VIDEO_URL: ${videoUrl}\n\n` +
        `PROMPT:\n${promptText}\n\n` +
        `Return JSON ONLY with keys:\n` +
        `emotional_tone (array)\nenergy (string)\nvisual_style (array)\ncontext (array)\n` +
        `human_presence (string)\nlighting (array)\ncolor_mood (array)\n`,
    },
    ...framesBase64.map((b64) => ({
      type: "input_image",
      image_url: `data:image/jpeg;base64,${b64}`,
    })),
  ];

  const resp = await openai.responses.create({
    model,
    input: [{ role: "user", content }],
  });

  const outText =
    resp.output_text ||
    resp.output?.[0]?.content?.map((c) => c.text).join("") ||
    "";

  return outText;
}

/**
 * STEP 10) ✅ NEW: parse “text JSON” safely
 * - handles junk lines like "Menu"
 * - extracts first {...} block and JSON.parse it
 */
function safeJsonFromText(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("OpenAI output did not contain a JSON object");
  }

  const jsonBlock = text.slice(start, end + 1);

  // remove obvious junk lines like "Menu" that can appear mid-object
  const cleaned = jsonBlock
    .split("\n")
    .filter((line) => line.trim() !== "Menu")
    .join("\n");

  return JSON.parse(cleaned);
}

/**
 * STEP 11) ✅ save to Airtable (FULLY DYNAMIC, TEXT ONLY)
 */
function toText(val) {
  if (Array.isArray(val)) return val.map(String).filter(Boolean).join(", ");
  if (val === null || val === undefined) return "";
  return String(val).trim();
}

async function createAirtableRow({ videoUrl, tagsObj }) {
  if (!AIRTABLE_API_KEY) throw new Error("Missing env var: AIRTABLE_API_KEY");
  if (!AIRTABLE_BASE_ID) throw new Error("Missing env var: AIRTABLE_BASE_ID");
  if (!AIRTABLE_TABLE_NAME) throw new Error("Missing env var: AIRTABLE_TABLE_NAME");

  const url = `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/${encodeURIComponent(
    AIRTABLE_TABLE_NAME
  )}`;

  const fields = { video_url: toText(videoUrl) };

  for (const [key, value] of Object.entries(tagsObj || {})) {
    const text = toText(value);
    if (!text) continue;
    fields[key] = text;
  }

  const body = { records: [{ fields }] };

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${AIRTABLE_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(`Airtable error: ${resp.status} ${JSON.stringify(data)}`);
  }
  return data;
}

// STEP 11.5) ✅ NEW: get ALL video_url values from Airtable (for diff)
async function getAllAirtableVideoUrls() {
  if (!AIRTABLE_API_KEY) throw new Error("Missing env var: AIRTABLE_API_KEY");
  if (!AIRTABLE_BASE_ID) throw new Error("Missing env var: AIRTABLE_BASE_ID");
  if (!AIRTABLE_TABLE_NAME) throw new Error("Missing env var: AIRTABLE_TABLE_NAME");

  const baseUrl = `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/${encodeURIComponent(
    AIRTABLE_TABLE_NAME
  )}`;

  const urls = [];
  let offset = undefined;

  while (true) {
    const url = new URL(baseUrl);
    url.searchParams.append("fields[]", "video_url");
    url.searchParams.set("pageSize", "100");
    if (offset) url.searchParams.set("offset", offset);

    const resp = await fetch(url.toString(), {
      headers: { Authorization: `Bearer ${AIRTABLE_API_KEY}` },
    });

    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(`Airtable read error: ${resp.status} ${JSON.stringify(data)}`);
    }

    for (const rec of data.records || []) {
      const v = rec?.fields?.video_url;
      if (v) urls.push(String(v).trim());
    }

    if (!data.offset) break;
    offset = data.offset;
  }

  return urls;
}

// STEP 12) health check
app.get("/", (_req, res) => {
  console.log("Health check hit");
  res.send("tagging service ok");
});

// STEP 13) webhook endpoint
app.post("/webhook", async (_req, res) => {
  let tmpDir = null;

  try {
    console.log("🔥 WEBHOOK CAPTURED (BATCH MODE)");

    const promptText = await getNotionPrompt();

    console.log("Listing ALL upload keys from R2...");
    const allKeys = await listAllUploadKeys();
    if (!allKeys.length) {
      console.log("⚠️ No files found in bucket");
      return res.status(404).json({ ok: false, error: "Bucket empty" });
    }
    console.log(`📦 Found ${allKeys.length} objects in R2`);

    const allUrls = allKeys.map(buildPublicVideoUrl);

    console.log("Fetching Airtable existing video_url values...");
    const existingUrls = await getAllAirtableVideoUrls();
    const existingSet = new Set(existingUrls);
    console.log(`📌 Airtable has ${existingUrls.length} video_url entries`);

    const newUrls = allUrls.filter((u) => !existingSet.has(u));
    console.log(`🆕 New URLs to tag: ${newUrls.length}`);

    const LIMIT = Number(process.env.BATCH_LIMIT || 10);
    // STEP: Randomize new videos before picking batch
    const shuffledNewUrls = [...newUrls].sort(() => Math.random() - 0.5);
    
    // STEP: Pick only the batch limit
    const toProcess = shuffledNewUrls.slice(0, LIMIT);

    const results = [];

    for (const videoUrl of toProcess) {
      console.log("====================================");
      console.log("🌐 PROCESSING:", videoUrl);

      try {
        console.log("⬇️ Downloading video...");
        const dl = await downloadToTempFile(videoUrl);
        tmpDir = dl.tmpDir;

        console.log("🖼️ Extracting frames...");
        const framesBase64 = await extractFramesBase64(dl.videoPath);
        console.log(`✅ Extracted ${framesBase64.length} frames`);

        console.log("🤖 Sending frames + prompt to OpenAI...");
        const analysisText = await analyzeVideoWithOpenAI({
          promptText,
          framesBase64,
          videoUrl,
        });

        // Help GC: drop big arrays ASAP
        framesBase64.length = 0;

        console.log("✅ OPENAI RAW TEXT:", analysisText);

        const tagsObj = safeJsonFromText(analysisText);

        console.log("📌 Creating Airtable row...");
        const airtableResp = await createAirtableRow({ videoUrl, tagsObj });
        console.log("✅ Airtable saved:", airtableResp?.records?.[0]?.id);

        results.push({
          videoUrl,
          ok: true,
          parsed: tagsObj,
          airtableRecordId: airtableResp?.records?.[0]?.id || null,
        });
      } catch (err) {
        console.error("❌ Failed processing:", videoUrl, err);
        results.push({
          videoUrl,
          ok: false,
          error: err.message,
        });
      } finally {
        if (tmpDir) {
          try {
            await fs.rm(tmpDir, { recursive: true, force: true });
          } catch {}
          tmpDir = null;
        }
      }
    }

    return res.json({
      ok: true,
      r2_total: allKeys.length,
      airtable_existing: existingUrls.length,
      new_found: newUrls.length,
      processed_now: toProcess.length,
      remaining_new: Math.max(0, newUrls.length - toProcess.length),
      results,
    });
  } catch (e) {
    console.error("Webhook error:", e);
    return res.status(500).json({ ok: false, error: e.message });
  }
});

// STEP 14) start server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
