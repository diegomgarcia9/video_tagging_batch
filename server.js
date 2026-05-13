import dotenv from "dotenv";
dotenv.config();

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
  PutObjectCommand,
  GetObjectCommand,
} from "@aws-sdk/client-s3";
import OpenAI from "openai";
import ffmpeg from "fluent-ffmpeg";
import multer from "multer";
import { execSync } from "child_process";

const app = express();

app.use(cors());
app.use(express.json({ limit: "1mb" }));

import ffprobeStatic from "ffprobe-static";
import { createClient } from "@supabase/supabase-js";

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);

// Use system ffmpeg (installed via apt-get during Render build) rather than
// ffmpeg-static, whose pre-compiled binary fails to run on Render's Linux env
// due to a GLIBC version mismatch.
ffmpeg.setFfmpegPath("ffmpeg");
ffmpeg.setFfprobePath("ffprobe");

try {
  const version = execSync("ffmpeg -version 2>&1", { timeout: 5000 }).toString().split("\n")[0];
  console.error(`[startup] ffmpeg version: ${version}`);
} catch (e) {
  console.error(`[startup] ffmpeg FAILED TO RUN — is ffmpeg installed? ${e.message}`);
}

const {
  CF_ACCOUNT_ID,
  R2_ACCESS_KEY_ID,
  R2_SECRET_ACCESS_KEY,
  R2_BUCKET_NAME,       // destination bucket: video-jobs-data
  R2_PUBLIC_BASE_URL,   // public URL for video-jobs-data
  OPENAI_API_KEY,
  OPENAI_VISION_MODEL,
} = process.env;

const REQUIRED_ENV = [
  "CF_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
  "R2_BUCKET_NAME", "R2_PUBLIC_BASE_URL",
  "OPENAI_API_KEY",
  "SUPABASE_URL", "SUPABASE_SERVICE_KEY",
];
for (const key of REQUIRED_ENV) {
  if (!process.env[key]) throw new Error(`Missing required env var: ${key}`);
}

const r2 = new S3Client({
  region: "auto",
  endpoint: `https://${CF_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: R2_ACCESS_KEY_ID,
    secretAccessKey: R2_SECRET_ACCESS_KEY,
  },
});

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

// ── FFmpeg helpers ────────────────────────────────────────────────────────────

async function downloadToTempFile(url) {
  console.error(`[download] fetching: ${url}`);
  const res = await fetch(url);
  const contentType = res.headers.get("content-type") || "unknown";
  console.error(`[download] status=${res.status} content-type=${contentType}`);
  if (!res.ok) throw new Error(`Failed to download video (${res.status})`);
  if (!contentType.includes("video") && !contentType.includes("octet-stream")) {
    const preview = (await res.text()).slice(0, 200);
    throw new Error(`Unexpected content-type "${contentType}". Body preview: ${preview}`);
  }
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
    const cmd = ffmpeg(input)
      .inputOptions(["-t 5"])
      .outputOptions([
        "-vf format=yuv420p,crop=if(gt(iw\\,ih)\\,ih*9/16\\,iw):if(gt(iw\\,ih)\\,ih\\,iw*16/9),scale=1080:1920,fps=30",
        "-c:v libx264",
        "-pix_fmt yuv420p",
        "-preset ultrafast",
        "-threads 1",
        "-an",
        "-movflags +faststart",
      ])
      .output(clipPath);

    cmd.on("start", (cmdLine) => console.error(`[ffmpeg cmd] ${cmdLine}`));
    cmd.on("stderr", (line) => console.error(`[ffmpeg] ${line}`));
    cmd.on("end", resolve);
    cmd.on("error", (err, _stdout, stderr) => {
      console.error(`[ffmpeg error] ${err.message}`);
      console.error(`[ffmpeg stderr dump] ${stderr}`);
      reject(err);
    });
    cmd.run();
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

async function getTaggingPromptText() {
  const bucket = process.env.R2_CONFIG_BUCKET_NAME || "config";
  const resp = await r2.send(new GetObjectCommand({ Bucket: bucket, Key: "prompts/tagging.json" }));
  const data = JSON.parse(await resp.Body.transformToString());

  const lines = [];
  if (data.role_block) lines.push(data.role_block, "");
  if (data.parameter_instructions) {
    lines.push("Field instructions:");
    for (const [key, instruction] of Object.entries(data.parameter_instructions)) {
      lines.push(`${key}: ${instruction}`);
    }
    lines.push("");
  }
  if (data.sequence_rules?.length) {
    lines.push("Rules:");
    for (const rule of data.sequence_rules) lines.push(`- ${rule}`);
    lines.push("");
  }
  if (data.goal) lines.push(`Goal: ${data.goal}`);
  return lines.join("\n");
}

async function analyzeVideoWithOpenAI({ promptText, framesBase64, videoUrl }) {
  const model = OPENAI_VISION_MODEL || "gpt-4.1-mini";

  const content = [
    {
      type: "input_text",
      text: `VIDEO_URL: ${videoUrl}\n\n${promptText}`,
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

// ── Supabase helpers ──────────────────────────────────────────────────────────

async function createSupabaseAsset({ videoUrl, thumbnailUrl = "", sourceUrl = "", status = "processing" }) {
  const { data, error } = await supabase
    .from("assets")
    .insert({
      video_url: videoUrl,
      thumbnail_url: thumbnailUrl || null,
      source_url: sourceUrl || null,
      status,
      ingested_at: new Date().toISOString(),
    })
    .select("id")
    .single();
  if (error) throw new Error(`Supabase create error: ${error.message}`);
  return data; // { id }
}

async function updateSupabaseStatus(assetId, status, extraFields = {}) {
  const { error } = await supabase
    .from("assets")
    .update({ status, ...extraFields })
    .eq("id", assetId);
  if (error) throw new Error(`Supabase status update error: ${error.message}`);
}

async function upsertSupabaseTags(assetId, tagsObj, model = "gpt-4o-mini") {
  await supabase
    .from("asset_tags")
    .update({ is_current: false })
    .eq("asset_id", assetId)
    .eq("is_current", true);

  const { error } = await supabase
    .from("asset_tags")
    .insert({
      asset_id: assetId,
      tags: tagsObj,
      tagged_with_params: Object.keys(tagsObj),
      tagged_by: model,
      is_current: true,
      tagged_at: new Date().toISOString(),
    });
  if (error) throw new Error(`Supabase tags insert error: ${error.message}`);
}

async function getSupabaseAsset(assetId) {
  const { data, error } = await supabase
    .from("assets")
    .select("id, video_url, thumbnail_url, source_url, status")
    .eq("id", assetId)
    .single();
  if (error) throw new Error(`Supabase get error: ${error.message}`);
  return data;
}

async function getAllSupabaseVideoUrls() {
  const { data, error } = await supabase.from("assets").select("video_url");
  if (error) throw new Error(`Supabase list error: ${error.message}`);
  return (data || []).map((r) => String(r.video_url || "").trim()).filter(Boolean);
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get("/", (_req, res) => res.send("tagging service ok"));

// POST /webhook — tag untagged clips from video-jobs-data/clips/
app.post("/webhook", async (_req, res) => {
  let tmpDir = null;

  try {
    const promptText = await getTaggingPromptText();

    const allKeys = await listAllUploadKeys(R2_BUCKET_NAME, "clips/");
    if (!allKeys.length) {
      return res.status(404).json({ ok: false, error: "No clips found in bucket" });
    }

    const allUrls = allKeys.map((k) => buildDestUrl(k));
    const existingUrls = await getAllSupabaseVideoUrls();
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

        airtableRecord = await createSupabaseAsset({ videoUrl, thumbnailUrl, status: "processing" });
        const recordId = airtableRecord.id;

        const dl = await downloadToTempFile(videoUrl);
        tmpDir = dl.tmpDir;

        const framesBase64 = await extractFramesBase64(dl.videoPath);
        const analysisText = await analyzeVideoWithOpenAI({ promptText, framesBase64, videoUrl });
        framesBase64.length = 0;

        const tagsObj = safeJsonFromText(analysisText);
        const cleanTags = {};
        for (const [key, value] of Object.entries(tagsObj || {})) {
          const text = toText(value);
          if (text) cleanTags[key] = text;
        }

        await upsertSupabaseTags(recordId, cleanTags, OPENAI_VISION_MODEL || "gpt-4o-mini");
        await updateSupabaseStatus(recordId, "tagged", { processed_at: new Date().toISOString() });

        results.push({ videoUrl, ok: true, parsed: tagsObj, assetId: recordId });
      } catch (err) {
        console.error("Failed processing:", videoUrl, err);
        if (airtableRecord?.id) {
          try {
            await updateSupabaseStatus(airtableRecord.id, "failed");
          } catch {}
        }
        results.push({ videoUrl, ok: false, assetId: airtableRecord?.id || null, error: err.message });
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
      supabase_existing: existingUrls.length,
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

// POST /retag/:recordId — re-run AI tagging on a specific clip
app.post("/retag/:recordId", async (req, res) => {
  let tmpDir = null;

  try {
    const { recordId } = req.params;
    const asset = await getSupabaseAsset(recordId);
    const videoUrl = asset?.video_url;
    if (!videoUrl) return res.status(400).json({ ok: false, error: "Asset has no video_url" });

    const promptText = await getTaggingPromptText();

    await updateSupabaseStatus(recordId, "processing");

    const dl = await downloadToTempFile(videoUrl);
    tmpDir = dl.tmpDir;

    const framesBase64 = await extractFramesBase64(dl.videoPath);
    const analysisText = await analyzeVideoWithOpenAI({ promptText, framesBase64, videoUrl });
    framesBase64.length = 0;

    const tagsObj = safeJsonFromText(analysisText);
    const cleanTags = {};
    for (const [key, value] of Object.entries(tagsObj || {})) {
      const text = toText(value);
      if (text) cleanTags[key] = text;
    }

    await upsertSupabaseTags(recordId, cleanTags, OPENAI_VISION_MODEL || "gpt-4o-mini");
    await updateSupabaseStatus(recordId, "tagged", { processed_at: new Date().toISOString() });

    return res.json({ ok: true, assetId: recordId, parsed: tagsObj });
  } catch (e) {
    console.error("Retag error:", e);
    if (req.params.recordId) {
      try { await updateSupabaseStatus(req.params.recordId, "failed"); } catch {}
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

    const rec = await createSupabaseAsset({
      videoUrl: newClipUrl,
      thumbnailUrl: newThumbUrl,
      status: "uploaded",
    });

    return res.json({
      ok: true,
      assetId: rec.id,
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
app.listen(port, () => {
  console.log(`Tagging service listening on port ${port}`);
});
