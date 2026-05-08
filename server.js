import express from "express";
import cors from "cors";
import fs from "fs/promises";
import { createReadStream, createWriteStream } from "fs";
import { pipeline } from "stream/promises";
import path from "path";
import os from "os";

import {
  S3Client,
  ListObjectsV2Command,
  HeadObjectCommand,
  PutObjectCommand,
} from "@aws-sdk/client-s3";
import { Client as NotionClient } from "@notionhq/client";
import OpenAI from "openai";
import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";
import multer from "multer";

const app = express();

app.use(cors());
app.use(express.json({ limit: "1mb" }));

import ffprobeStatic from "ffprobe-static";

ffmpeg.setFfmpegPath(ffmpegPath);
ffmpeg.setFfprobePath(ffprobeStatic.path);

const {
  R2_ACCOUNT_ID,
  R2_ACCESS_KEY_ID,
  R2_SECRET_ACCESS_KEY,
  R2_BUCKET_NAME,       // destination bucket: video-jobs-data
  R2_PUBLIC_BASE_URL,   // public URL for video-jobs-data

  SOURCE_BUCKET,        // migration source: content-machine
  SOURCE_PUBLIC_URL,    // public URL for content-machine

  NOTION_API_KEY,
  NOTION_PAGE_ID,

  OPENAI_API_KEY,
  OPENAI_VISION_MODEL,

  AIRTABLE_API_KEY,
  AIRTABLE_BASE_ID,
  AIRTABLE_TABLE_NAME,
} = process.env;

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

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, os.tmpdir()),
    filename: (_req, file, cb) => cb(null, `upload_${Date.now()}_${file.originalname}`),
  }),
  limits: { fileSize: 500 * 1024 * 1024 },
});

// ── Tag fields used to determine completeness ────────────────────────────────
const TAG_FIELDS = [
  "emotional_tone",
  "energy_level",
  "visual_style",
  "context",
  "human_presence",
  "lighting",
  "color_mood",
  "background",
  "pace",
  "narrative_function",
  "visual_medium",
  "short_description",
];

// ── R2 helpers ───────────────────────────────────────────────────────────────

async function listAllUploadKeys(bucket, prefix = "") {
  const keys = [];
  let ContinuationToken;

  while (true) {
    const command = new ListObjectsV2Command({
      Bucket: bucket,
      Prefix: prefix,
      MaxKeys: 1000,
      ContinuationToken,
    });

    const response = await r2.send(command);

    for (const f of response.Contents || []) {
      if (f?.Key && !f.Key.endsWith("/")) keys.push(f.Key);
    }

    if (!response.IsTruncated) break;
    ContinuationToken = response.NextContinuationToken;
  }

  return keys;
}

async function doesKeyExist(bucket, key) {
  try {
    await r2.send(new HeadObjectCommand({ Bucket: bucket, Key: key }));
    return true;
  } catch {
    return false;
  }
}

async function uploadFileToR2(bucket, key, filePath, contentType) {
  const stream = createReadStream(filePath);
  await r2.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: stream,
      ContentType: contentType,
    })
  );
}

function buildPublicUrl(baseUrl, key) {
  const base = baseUrl.replace(/\/+$/, "");
  const safeKey = key
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
  return `${base}/${safeKey}`;
}

function buildDestUrl(key) {
  return buildPublicUrl(R2_PUBLIC_BASE_URL, key);
}

function buildSourceUrl(key) {
  return buildPublicUrl(SOURCE_PUBLIC_URL, key);
}

// ── FFmpeg helpers ────────────────────────────────────────────────────────────

async function downloadToTempFile(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download video (${res.status})`);
  if (!res.body) throw new Error("No response body to stream");

  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "video-"));
  const videoPath = path.join(tmpDir, "input.mp4");
  await pipeline(res.body, createWriteStream(videoPath));

  return { tmpDir, videoPath };
}

// Standardize to 30fps, trim to 5 seconds, extract thumbnail at frame 50 (≈1.667s).
// `input` can be a local file path or a public URL — FFmpeg reads directly from URLs,
// so for remote files we never download more than 5 seconds of data.
async function processVideoForStorage(input, outputDir) {
  const clipPath = path.join(outputDir, "clip.mp4");
  const thumbPath = path.join(outputDir, "thumb.jpg");

  // Center-crop to 9:16 then scale to 1080x1920.
  // -cpuflags 0 disables all SIMD (AVX2/SSE) optimizations. ffmpeg-static has a
  // known alignment SIGSEGV with 4096-wide frames because 4096 = 2^12 triggers
  // edge cases in vectorized decode routines. Plain-C fallbacks are safe.
  // format=yuv420p as first filter immediately converts 10-bit HDR frames to 8-bit,
  // halving per-frame memory before any other operation runs.
  await new Promise((resolve, reject) => {
    ffmpeg(input)
      .inputOptions(["-cpuflags 0", "-t 5"])
      .outputOptions([
        "-vf format=yuv420p,crop=if(gt(iw\\,ih)\\,ih*9/16\\,iw):if(gt(iw\\,ih)\\,ih\\,iw*16/9),scale=1080:1920,fps=30",
        "-c:v libx264",
        "-pix_fmt yuv420p",
        "-preset ultrafast",
        "-threads 1",
        "-an",
        "-movflags +faststart",
      ])
      .output(clipPath)
      .on("end", resolve)
      .on("error", reject)
      .run();
  });

  // Extract thumbnail from the already-processed clip (1080x1920, ~few MB) —
  // never decode the 4K source a second time, which is what caused SIGSEGV.
  await new Promise((resolve, reject) => {
    ffmpeg(clipPath)
      .inputOptions(["-ss 1.667"])
      .outputOptions(["-frames:v 1", "-q:v 2"])
      .output(thumbPath)
      .on("end", resolve)
      .on("error", reject)
      .run();
  });

  return { clipPath, thumbPath };
}

// Probe a video (local path or URL) and return the first video stream's metadata.
// Throws if ffprobe cannot read the file — caller should skip/log gracefully.
async function probeVideo(input) {
  return new Promise((resolve, reject) => {
    // -cpuflags 0 disables SIMD in ffprobe for the same reason we disable it in
    // ffmpeg: the 2018-era static binary crashes with SIGSEGV on 4096-wide frames
    // due to AVX2/SSE alignment assumptions that don't hold at power-of-2 widths.
    ffmpeg.ffprobe(input, ["-cpuflags", "0"], (err, metadata) => {
      if (err) return reject(err);
      const videoStream = (metadata.streams || []).find((s) => s.codec_type === "video");
      if (!videoStream) return reject(new Error("No video stream found"));
      resolve({
        codec: videoStream.codec_name,
        pixFmt: videoStream.pix_fmt,
        width: videoStream.width,
        height: videoStream.height,
        duration: metadata.format?.duration,
      });
    });
  });
}

async function extractFramesBase64(videoPath, { fps = 1, maxFrames = 4 } = {}) {
  const framesDir = path.join(path.dirname(videoPath), "frames");
  await fs.mkdir(framesDir, { recursive: true });

  await new Promise((resolve, reject) => {
    ffmpeg(videoPath)
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
    const b = await fs.readFile(path.join(framesDir, f));
    images.push(b.toString("base64"));
  }

  return images;
}

// ── OpenAI ────────────────────────────────────────────────────────────────────

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

  if (!textParts) return `[Prompt property type "${prop?.type ?? "unknown"}" not supported]`;
  return textParts.map((t) => t.plain_text).join("").trim() || "[Prompt is empty]";
}

async function analyzeVideoWithOpenAI({ promptText, framesBase64, videoUrl }) {
  const model = OPENAI_VISION_MODEL || "gpt-4.1-mini";

  const content = [
    {
      type: "input_text",
      text:
        `Analyze this stock video for retrieval.\n` +
        `VIDEO_URL: ${videoUrl}\n\n` +
        `PROMPT:\n${promptText}\n\n` +
        `Return JSON ONLY with keys:\n` +
        `emotional_tone (array)\n` +
        `energy (string)\n` +
        `visual_style (array)\n` +
        `context (array)\n` +
        `human_presence (string)\n` +
        `lighting (array)\n` +
        `color_mood (array)\n`,
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

  return (
    resp.output_text ||
    resp.output?.[0]?.content?.map((c) => c.text).join("") ||
    ""
  );
}

function safeJsonFromText(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("OpenAI output did not contain a JSON object");
  }
  const cleaned = text
    .slice(start, end + 1)
    .split("\n")
    .filter((line) => line.trim() !== "Menu")
    .join("\n");
  return JSON.parse(cleaned);
}

function toText(val) {
  if (Array.isArray(val)) return val.map(String).filter(Boolean).join(", ");
  if (val === null || val === undefined) return "";
  return String(val).trim();
}

// ── Airtable helpers ──────────────────────────────────────────────────────────

function airtableUrl(path = "") {
  return `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/${encodeURIComponent(AIRTABLE_TABLE_NAME)}${path}`;
}

function airtableHeaders() {
  return {
    Authorization: `Bearer ${AIRTABLE_API_KEY}`,
    "Content-Type": "application/json",
  };
}

async function createAirtableRow({ videoUrl, thumbnailUrl = "", status = "processing" }) {
  const body = {
    records: [
      {
        fields: {
          video_url: toText(videoUrl),
          ...(thumbnailUrl ? { thumbnail_url: toText(thumbnailUrl) } : {}),
          status,
        },
      },
    ],
  };

  const resp = await fetch(airtableUrl(), {
    method: "POST",
    headers: airtableHeaders(),
    body: JSON.stringify(body),
  });

  const data = await resp.json();
  if (!resp.ok) throw new Error(`Airtable create error: ${resp.status} ${JSON.stringify(data)}`);
  return data?.records?.[0];
}

async function updateAirtableRow({ recordId, fields }) {
  const body = { records: [{ id: recordId, fields }] };
  const resp = await fetch(airtableUrl(), {
    method: "PATCH",
    headers: airtableHeaders(),
    body: JSON.stringify(body),
  });

  const data = await resp.json();
  if (!resp.ok) throw new Error(`Airtable update error: ${resp.status} ${JSON.stringify(data)}`);
  return data?.records?.[0];
}

async function getAirtableRecord(recordId) {
  const resp = await fetch(airtableUrl(`/${recordId}`), {
    headers: { Authorization: `Bearer ${AIRTABLE_API_KEY}` },
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(`Airtable get error: ${resp.status} ${JSON.stringify(data)}`);
  return data;
}

// Returns all records as array of { id, fields }
async function getAllAirtableRecords(fieldsToFetch = []) {
  const records = [];
  let offset;

  while (true) {
    const url = new URL(airtableUrl());
    url.searchParams.set("pageSize", "100");
    for (const f of fieldsToFetch) url.searchParams.append("fields[]", f);
    if (offset) url.searchParams.set("offset", offset);

    const resp = await fetch(url.toString(), {
      headers: { Authorization: `Bearer ${AIRTABLE_API_KEY}` },
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(`Airtable list error: ${resp.status} ${JSON.stringify(data)}`);

    for (const rec of data.records || []) {
      records.push({ id: rec.id, fields: rec.fields });
    }

    if (!data.offset) break;
    offset = data.offset;
  }

  return records;
}

async function getAllAirtableVideoUrls() {
  const records = await getAllAirtableRecords(["video_url"]);
  return records
    .map((r) => String(r.fields.video_url || "").trim())
    .filter(Boolean);
}

// ── Status determination ──────────────────────────────────────────────────────

function computeStatus(fields) {
  const current = String(fields.status || "").toLowerCase();
  // Don't overwrite these states — they're set by active processes
  if (current === "processing") return null;

  const filledTags = TAG_FIELDS.filter((f) => {
    const val = fields[f];
    return val && String(val).trim().length > 0;
  });

  if (filledTags.length === TAG_FIELDS.length) return "tagged";
  if (filledTags.length > 0) return "incomplete";
  if (fields.video_url) return "uploaded";
  return null;
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get("/", (_req, res) => res.send("tagging service ok"));

// POST /webhook — tag untagged clips from video-jobs-data/clips/
app.post("/webhook", async (_req, res) => {
  let tmpDir = null;

  try {
    const promptText = await getNotionPrompt();

    const allKeys = await listAllUploadKeys(R2_BUCKET_NAME, "clips/");
    if (!allKeys.length) {
      return res.status(404).json({ ok: false, error: "No clips found in bucket" });
    }

    const allUrls = allKeys.map((k) => buildDestUrl(k));
    const existingUrls = await getAllAirtableVideoUrls();
    const existingSet = new Set(existingUrls);
    const newUrls = allUrls.filter((u) => !existingSet.has(u));

    const LIMIT = Number(process.env.BATCH_LIMIT || 10);
    const toProcess = [...newUrls].sort(() => Math.random() - 0.5).slice(0, LIMIT);
    const results = [];

    for (const videoUrl of toProcess) {
      let airtableRecord = null;

      try {
        // Derive thumbnail URL from clip key
        const clipKey = toProcess.indexOf(videoUrl) >= 0
          ? allKeys[allUrls.indexOf(videoUrl)]
          : null;
        const basename = clipKey ? path.basename(clipKey, path.extname(clipKey)) : null;
        const thumbnailUrl = basename ? buildDestUrl(`thumbnails/${basename}.jpg`) : "";

        airtableRecord = await createAirtableRow({ videoUrl, thumbnailUrl, status: "processing" });
        const recordId = airtableRecord.id;

        const dl = await downloadToTempFile(videoUrl);
        tmpDir = dl.tmpDir;

        const framesBase64 = await extractFramesBase64(dl.videoPath);
        const analysisText = await analyzeVideoWithOpenAI({ promptText, framesBase64, videoUrl });
        framesBase64.length = 0;

        const tagsObj = safeJsonFromText(analysisText);
        const updateFields = { status: "tagged" };
        for (const [key, value] of Object.entries(tagsObj || {})) {
          const text = toText(value);
          if (text) updateFields[key] = text;
        }

        await updateAirtableRow({ recordId, fields: updateFields });

        results.push({ videoUrl, ok: true, parsed: tagsObj, airtableRecordId: recordId });
      } catch (err) {
        console.error("Failed processing:", videoUrl, err);
        if (airtableRecord?.id) {
          try {
            await updateAirtableRow({
              recordId: airtableRecord.id,
              fields: { status: "failed", error_message: String(err.message).slice(0, 1000) },
            });
          } catch {}
        }
        results.push({ videoUrl, ok: false, airtableRecordId: airtableRecord?.id || null, error: err.message });
      } finally {
        if (tmpDir) {
          try { await fs.rm(tmpDir, { recursive: true, force: true }); } catch {}
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

// POST /migrate — process clips from content-machine into video-jobs-data/clips/
app.post("/migrate", async (req, res) => {
  let tmpDir = null;

  try {
    if (!SOURCE_BUCKET) return res.status(400).json({ ok: false, error: "SOURCE_BUCKET env var not set" });
    if (!SOURCE_PUBLIC_URL) return res.status(400).json({ ok: false, error: "SOURCE_PUBLIC_URL env var not set" });

    const LIMIT = Number(req.body?.batch_size || process.env.MIGRATE_BATCH_LIMIT || 20);

    // List source and destination
    const sourceKeys = await listAllUploadKeys(SOURCE_BUCKET);
    const destClipKeys = await listAllUploadKeys(R2_BUCKET_NAME, "clips/");
    const processedFilenames = new Set(destClipKeys.map((k) => path.basename(k)));

    const unprocessed = sourceKeys.filter((k) => !processedFilenames.has(path.basename(k)));
    const toProcess = unprocessed.slice(0, LIMIT);

    // Load all Airtable records indexed by video_url
    const allRecords = await getAllAirtableRecords(["video_url", "thumbnail_url"]);
    const recordByUrl = new Map();
    for (const rec of allRecords) {
      if (rec.fields.video_url) recordByUrl.set(String(rec.fields.video_url).trim(), rec.id);
    }

    const results = [];

    for (const sourceKey of toProcess) {
      tmpDir = null;
      const filename = path.basename(sourceKey);
      const basename = path.basename(sourceKey, path.extname(sourceKey));

      const oldUrl = buildSourceUrl(sourceKey);
      const newClipKey = `clips/${filename}`;
      const newThumbKey = `thumbnails/${basename}.jpg`;
      const newClipUrl = buildDestUrl(newClipKey);
      const newThumbUrl = buildDestUrl(newThumbKey);

      try {
        // FFmpeg reads directly from the source URL — no full download needed.
        // The -t 5 input option means only ~5s of data is ever pulled from the source.
        tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "migrate-"));
        const { clipPath, thumbPath } = await processVideoForStorage(oldUrl, tmpDir);

        await uploadFileToR2(R2_BUCKET_NAME, newClipKey, clipPath, "video/mp4");
        await uploadFileToR2(R2_BUCKET_NAME, newThumbKey, thumbPath, "image/jpeg");

        // Find existing Airtable record by old URL, new URL, or create
        const existingByOld = recordByUrl.get(oldUrl);
        const existingByNew = recordByUrl.get(newClipUrl);

        if (existingByOld) {
          await updateAirtableRow({
            recordId: existingByOld,
            fields: { video_url: newClipUrl, thumbnail_url: newThumbUrl },
          });
          recordByUrl.delete(oldUrl);
          recordByUrl.set(newClipUrl, existingByOld);
          results.push({ filename, ok: true, action: "updated", airtableRecordId: existingByOld });
        } else if (existingByNew) {
          await updateAirtableRow({
            recordId: existingByNew,
            fields: { thumbnail_url: newThumbUrl },
          });
          results.push({ filename, ok: true, action: "thumbnail_updated", airtableRecordId: existingByNew });
        } else {
          const rec = await createAirtableRow({
            videoUrl: newClipUrl,
            thumbnailUrl: newThumbUrl,
            status: "uploaded",
          });
          recordByUrl.set(newClipUrl, rec.id);
          results.push({ filename, ok: true, action: "created", airtableRecordId: rec.id });
        }
      } catch (err) {
        console.error("Migration failed for:", sourceKey, err);
        results.push({ filename, ok: false, error: err.message });
      } finally {
        if (tmpDir) {
          try { await fs.rm(tmpDir, { recursive: true, force: true }); } catch {}
          tmpDir = null;
        }
      }
    }

    return res.json({
      ok: true,
      source_total: sourceKeys.length,
      already_processed: sourceKeys.length - unprocessed.length,
      new_found: unprocessed.length,
      processed_now: toProcess.length,
      remaining: Math.max(0, unprocessed.length - toProcess.length),
      results,
    });
  } catch (e) {
    console.error("Migration error:", e);
    return res.status(500).json({ ok: false, error: e.message });
  }
});

// POST /sync-statuses — recalculate status for all records based on tag field completeness
app.post("/sync-statuses", async (_req, res) => {
  try {
    const allRecords = await getAllAirtableRecords(["video_url", "status", ...TAG_FIELDS]);

    let updated = 0;
    let skipped = 0;

    // Batch updates (Airtable allows up to 10 per PATCH)
    const batchSize = 10;
    const updates = [];

    for (const rec of allRecords) {
      const newStatus = computeStatus(rec.fields);
      if (newStatus === null || newStatus === rec.fields.status) {
        skipped++;
        continue;
      }
      updates.push({ id: rec.id, fields: { status: newStatus } });
    }

    for (let i = 0; i < updates.length; i += batchSize) {
      const batch = updates.slice(i, i + batchSize);
      const body = { records: batch };
      const resp = await fetch(airtableUrl(), {
        method: "PATCH",
        headers: airtableHeaders(),
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(`Airtable batch update error: ${resp.status} ${JSON.stringify(data)}`);
      updated += batch.length;
    }

    return res.json({ ok: true, total: allRecords.length, updated, skipped });
  } catch (e) {
    console.error("Sync statuses error:", e);
    return res.status(500).json({ ok: false, error: e.message });
  }
});

// POST /retag/:recordId — re-run AI tagging on a specific clip
app.post("/retag/:recordId", async (req, res) => {
  let tmpDir = null;

  try {
    const { recordId } = req.params;
    const rec = await getAirtableRecord(recordId);
    const videoUrl = rec?.fields?.video_url;
    if (!videoUrl) return res.status(400).json({ ok: false, error: "Record has no video_url" });

    const promptText = await getNotionPrompt();

    await updateAirtableRow({ recordId, fields: { status: "processing" } });

    const dl = await downloadToTempFile(videoUrl);
    tmpDir = dl.tmpDir;

    const framesBase64 = await extractFramesBase64(dl.videoPath);
    const analysisText = await analyzeVideoWithOpenAI({ promptText, framesBase64, videoUrl });
    framesBase64.length = 0;

    const tagsObj = safeJsonFromText(analysisText);
    const updateFields = { status: "tagged" };
    for (const [key, value] of Object.entries(tagsObj || {})) {
      const text = toText(value);
      if (text) updateFields[key] = text;
    }

    await updateAirtableRow({ recordId, fields: updateFields });

    return res.json({ ok: true, airtableRecordId: recordId, parsed: tagsObj });
  } catch (e) {
    console.error("Retag error:", e);
    if (req.params.recordId) {
      try {
        await updateAirtableRow({
          recordId: req.params.recordId,
          fields: { status: "failed", error_message: String(e.message).slice(0, 1000) },
        });
      } catch {}
    }
    return res.status(500).json({ ok: false, error: e.message });
  } finally {
    if (tmpDir) {
      try { await fs.rm(tmpDir, { recursive: true, force: true }); } catch {}
    }
  }
});

// POST /upload — accept a video file, process it, upload to R2, create Airtable record
app.post("/upload", upload.single("clip"), async (req, res) => {
  let tmpDir = null;
  const uploadedPath = req.file?.path;

  try {
    if (!req.file) return res.status(400).json({ ok: false, error: "No file uploaded (field name: clip)" });

    const originalName = req.file.originalname.replace(/[^a-zA-Z0-9._-]/g, "_");
    const basename = path.basename(originalName, path.extname(originalName));
    const timestamp = Date.now();
    const clipFilename = `${basename}_${timestamp}.mp4`;
    const thumbFilename = `${basename}_${timestamp}.jpg`;

    const newClipKey = `clips/${clipFilename}`;
    const newThumbKey = `thumbnails/${thumbFilename}`;

    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "upload-"));
    const { clipPath, thumbPath } = await processVideoForStorage(uploadedPath, tmpDir);

    // Delete source upload before streaming processed files to R2
    try { await fs.unlink(uploadedPath); } catch {}

    await uploadFileToR2(R2_BUCKET_NAME, newClipKey, clipPath, "video/mp4");
    await uploadFileToR2(R2_BUCKET_NAME, newThumbKey, thumbPath, "image/jpeg");

    const newClipUrl = buildDestUrl(newClipKey);
    const newThumbUrl = buildDestUrl(newThumbKey);

    const rec = await createAirtableRow({
      videoUrl: newClipUrl,
      thumbnailUrl: newThumbUrl,
      status: "uploaded",
    });

    return res.json({
      ok: true,
      airtableRecordId: rec.id,
      video_url: newClipUrl,
      thumbnail_url: newThumbUrl,
    });
  } catch (e) {
    console.error("Upload error:", e);
    return res.status(500).json({ ok: false, error: e.message });
  } finally {
    if (uploadedPath) {
      try { await fs.unlink(uploadedPath); } catch {}
    }
    if (tmpDir) {
      try { await fs.rm(tmpDir, { recursive: true, force: true }); } catch {}
    }
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Tagging service listening on port ${port}`));
