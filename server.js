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
  "short_description",
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
  "camera_behavior",
  "shot_type",
  "movement_type",
  "keywords",
  "interaction_type",
  "action_type",
  "emotional_valence",
  "psychological_state",
  "emotional_intensity",
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

async function getParameters() {
  try {
    const bucket = process.env.R2_CONFIG_BUCKET_NAME || "config";
    const resp = await r2.send(new GetObjectCommand({ Bucket: bucket, Key: "parameters.json" }));
    const data = JSON.parse(await resp.Body.transformToString());
    return (data.parameters || []).filter((p) => !p.hidden);
  } catch {
    return [];
  }
}

async function getTaggingPromptText(videoUrl, parameters) {
  const bucket = process.env.R2_CONFIG_BUCKET_NAME || "config";
  const resp = await r2.send(new GetObjectCommand({ Bucket: bucket, Key: "prompts/tagging.json" }));
  const p = JSON.parse(await resp.Body.transformToString());

  function countLabel(param) {
    if (!param || param.input_type === "text") return "";
    if (param.input_type === "single") return "Single value.";
    if (param.min_count != null && param.max_count != null) return `${param.min_count}–${param.max_count} values.`;
    return "";
  }

  const instrLines = Object.entries(p.parameter_instructions || {}).map(([k, v]) => {
    const param = parameters.find((pr) => pr.name === k);
    const count = countLabel(param);
    const allowed = param?.allowed_values?.length
      ? `\n  Allowed: [${param.allowed_values.map((av) => `"${av}"`).join(", ")}]`
      : "";
    const instruction = [count, v].filter(Boolean).join(" ");
    return `${k}: ${instruction}${allowed}`;
  }).join("\n\n");

  const seqRules = (p.sequence_rules || []).map((r, i) => `${i + 1}. ${r}`).join("\n");

  const schemaEntries = parameters.length
    ? parameters.map((pr) => `  "${pr.name}": ${pr.input_type === "array" ? "[]" : '""'}`).join(",\n")
    : Object.keys(p.parameter_instructions || {}).map((k) => `  "${k}": ""`).join(",\n");

  return [
    p.role_block || "",
    instrLines ? `\nInstructions per field:\n\n${instrLines}` : "",
    seqRules ? `\nFormatting rules:\n${seqRules}` : "",
    p.goal ? `\nGoal: ${p.goal}` : "",
    "\n---",
    `VIDEO_URL: ${videoUrl}`,
    "[Frames attached as images]",
    `\n// Schema — fill every key, follow allowed values exactly:\n{\n${schemaEntries}\n}`,
  ].filter(Boolean).join("\n");
}

async function analyzeVideoWithOpenAI({ promptText, framesBase64 }) {
  const model = OPENAI_VISION_MODEL || "gpt-4o-mini";

  const content = [
    { type: "text", text: promptText },
    ...framesBase64.map((b64) => ({
      type: "image_url",
      image_url: { url: `data:image/jpeg;base64,${b64}` },
    })),
  ];

  const resp = await openai.chat.completions.create({
    model,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: "You are a video clip tagger. Return only valid JSON matching the schema exactly." },
      { role: "user", content },
    ],
  });

  return resp.choices?.[0]?.message?.content?.trim() || "";
}

async function correctTagResponseWithOpenAI(feedbackPrompt) {
  const model = OPENAI_VISION_MODEL || "gpt-4o-mini";
  const resp = await openai.chat.completions.create({
    model,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: "You are a JSON corrector. Return only valid JSON matching the schema." },
      { role: "user", content: feedbackPrompt },
    ],
  });
  return resp.choices?.[0]?.message?.content?.trim() || "";
}

function validateTagResponse({ tagsObj, parameters }) {
  const errors = [];
  for (const param of parameters) {
    const { name, input_type, allowed_values, min_count, max_count } = param;
    const val = tagsObj[name];

    if (name === "short_description") {
      if (!val || typeof val !== "string" || !val.trim()) {
        errors.push({ field: name, message: "short_description must be a non-empty string." });
      } else {
        const words = val.trim().split(/\s+/);
        if (words.length > 12) {
          errors.push({ field: name, message: `short_description must be max 12 words (got ${words.length}).` });
        }
      }
      continue;
    }

    if (input_type === "array") {
      const arr = Array.isArray(val) ? val : (typeof val === "string" && val ? [val] : []);
      const min = min_count ?? 1;
      const max = max_count ?? 10;
      if (arr.length < min || arr.length > max) {
        errors.push({ field: name, message: `${name} must contain ${min}–${max} values (got ${arr.length}).` });
      }
      if (allowed_values?.length) {
        for (const item of arr) {
          if (!allowed_values.includes(item)) {
            errors.push({ field: name, message: `${name} contains invalid value "${item}". Allowed: ${allowed_values.join(", ")}` });
          }
        }
      }
    } else {
      if (!val || typeof val !== "string" || !val.trim()) {
        errors.push({ field: name, message: `${name} must be a non-empty string.` });
      } else if (allowed_values?.length && !allowed_values.includes(val)) {
        errors.push({ field: name, message: `${name} contains invalid value "${val}". Allowed: ${allowed_values.join(", ")}` });
      }
    }
  }
  return { ok: errors.length === 0, errors };
}

function buildTaggingFeedback({ errors, previousResponse }) {
  const byField = {};
  for (const e of errors) {
    if (!byField[e.field]) byField[e.field] = [];
    byField[e.field].push(e.message);
  }
  const fieldErrors = Object.entries(byField)
    .map(([field, msgs]) => `- ${field}: ${msgs.join("; ")}`)
    .join("\n");
  return [
    "Your previous response failed validation.",
    "",
    "Fix the JSON and return the corrected JSON only.",
    "",
    "Fields with errors:",
    fieldErrors,
    "",
    "Here is your previous response. Correct it and return only the fixed JSON:",
    previousResponse,
  ].join("\n");
}

async function runTaggingWithRetry({ videoUrl, framesBase64, parameters }) {
  const promptText = await getTaggingPromptText(videoUrl, parameters);
  let responseText = "";
  let tagsObj = null;
  let validation = null;
  const maxAttempts = 3;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    if (attempt === 1) {
      responseText = await analyzeVideoWithOpenAI({ promptText, framesBase64 });
    } else {
      const feedbackPrompt = buildTaggingFeedback({ errors: validation.errors, previousResponse: responseText });
      responseText = await correctTagResponseWithOpenAI(feedbackPrompt);
    }

    try {
      tagsObj = JSON.parse(responseText);
    } catch (e) {
      validation = { ok: false, errors: [{ field: "json", message: `Invalid JSON: ${e.message}` }] };
      continue;
    }

    validation = validateTagResponse({ tagsObj, parameters });
    if (validation.ok) break;
  }

  return { tagsObj: tagsObj || {}, validation };
}

// ── Supabase helpers ──────────────────────────────────────────────────────────

function computeTagStatus(tags, parameters) {
  const visible = parameters.filter((p) => !p.hidden);
  if (!visible.length) {
    const hasAny = Object.values(tags).some((v) => Array.isArray(v) ? v.length > 0 : Boolean(v));
    return hasAny ? "incomplete" : "processed";
  }
  const hasAny = visible.some((p) => {
    const v = tags[p.name];
    return Array.isArray(v) ? v.length > 0 : Boolean(v);
  });
  if (!hasAny) return "processed";

  // Any filled field with an out-of-vocabulary value → invalid
  const hasInvalid = visible.some((p) => {
    if (!p.allowed_values?.length) return false;
    const val = tags[p.name];
    if (p.input_type === "array") {
      const arr = Array.isArray(val) ? val : (typeof val === "string" && val ? [val] : []);
      return arr.length > 0 && arr.some((item) => !p.allowed_values.includes(item));
    }
    if (p.input_type !== "text") {
      return val && typeof val === "string" && !p.allowed_values.includes(val.trim());
    }
    return false;
  });
  if (hasInvalid) return "invalid";

  const hasAll = visible.every((p) => {
    const v = tags[p.name];
    return Array.isArray(v) ? v.length > 0 : Boolean(v);
  });
  return hasAll ? "tagged" : "incomplete";
}

async function createSupabaseAsset({ videoUrl, thumbnailUrl = "", sourceUrl = "", status = "ingested" }) {
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
    const parameters = await getParameters();
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

        airtableRecord = await createSupabaseAsset({ videoUrl, thumbnailUrl, status: "tagging" });
        const recordId = airtableRecord.id;

        const dl = await downloadToTempFile(videoUrl);
        tmpDir = dl.tmpDir;

        const framesBase64 = await extractFramesBase64(dl.videoPath);
        const { tagsObj } = await runTaggingWithRetry({ videoUrl, framesBase64, parameters });
        framesBase64.length = 0;

        await upsertSupabaseTags(recordId, tagsObj, OPENAI_VISION_MODEL || "gpt-4o-mini");
        const tagStatus = computeTagStatus(tagsObj, parameters);
        await updateSupabaseStatus(recordId, tagStatus, { processed_at: new Date().toISOString() });

        results.push({ videoUrl, ok: true, tags: tagsObj, assetId: recordId });
      } catch (err) {
        console.error("Failed processing:", videoUrl, err);
        if (airtableRecord?.id) {
          try {
            await updateSupabaseStatus(airtableRecord.id, "failed_tagging");
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

    const parameters = await getParameters();

    await updateSupabaseStatus(recordId, "tagging");

    const dl = await downloadToTempFile(videoUrl);
    tmpDir = dl.tmpDir;

    const framesBase64 = await extractFramesBase64(dl.videoPath);
    const { tagsObj } = await runTaggingWithRetry({ videoUrl, framesBase64, parameters });
    framesBase64.length = 0;

    await upsertSupabaseTags(recordId, tagsObj, OPENAI_VISION_MODEL || "gpt-4o-mini");
    const tagStatus = computeTagStatus(tagsObj, parameters);
    await updateSupabaseStatus(recordId, tagStatus, { processed_at: new Date().toISOString() });

    return res.json({ ok: true, assetId: recordId, tags: tagsObj });
  } catch (e) {
    console.error("Retag error:", e);
    if (req.params.recordId) {
      try { await updateSupabaseStatus(req.params.recordId, "failed_tagging"); } catch {}
    }
    return res.status(500).json({ ok: false, error: e.message });
  } finally {
    if (tmpDir) {
      try { await fs.rm(tmpDir, { recursive: true, force: true }); } catch {}
    }
  }
});

// POST /sync-statuses — re-evaluate status for all tagged/incomplete/invalid assets
app.post("/sync-statuses", async (_req, res) => {
  try {
    const parameters = await getParameters();

    const { data: assets, error: listErr } = await supabase
      .from("assets")
      .select("id, status")
      .in("status", ["tagged", "incomplete", "invalid"]);
    if (listErr) throw new Error(`Supabase list error: ${listErr.message}`);

    if (!assets.length) return res.json({ ok: true, checked: 0, updated: 0 });

    const ids = assets.map((a) => a.id);
    const { data: tagRows, error: tagsErr } = await supabase
      .from("asset_tags")
      .select("asset_id, tags")
      .in("asset_id", ids)
      .eq("is_current", true);
    if (tagsErr) throw new Error(`Supabase tags error: ${tagsErr.message}`);

    const tagMap = {};
    for (const row of tagRows || []) tagMap[row.asset_id] = row.tags || {};

    const byStatus = {};
    let updatedCount = 0;
    for (const asset of assets) {
      const newStatus = computeTagStatus(tagMap[asset.id] || {}, parameters);
      if (newStatus !== asset.status) {
        if (!byStatus[newStatus]) byStatus[newStatus] = [];
        byStatus[newStatus].push(asset.id);
        updatedCount++;
      }
    }

    for (const [status, statusIds] of Object.entries(byStatus)) {
      const { error: upErr } = await supabase.from("assets").update({ status }).in("id", statusIds);
      if (upErr) throw new Error(`Supabase update error: ${upErr.message}`);
    }

    return res.json({ ok: true, checked: assets.length, updated: updatedCount });
  } catch (e) {
    console.error("Sync-statuses error:", e);
    return res.status(500).json({ ok: false, error: e.message });
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

    // Create asset record immediately so the clip is visible in the UI
    const newClipUrl = buildDestUrl(newClipKey);
    const newThumbUrl = buildDestUrl(newThumbKey);
    const rec = await createSupabaseAsset({
      videoUrl: newClipUrl,
      thumbnailUrl: newThumbUrl,
      status: "ingested",
    });

    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "upload-"));
    let clipPath, thumbPath;
    try {
      ({ clipPath, thumbPath } = await processVideoForStorage(uploadedPath, tmpDir));
    } catch (procErr) {
      await updateSupabaseStatus(rec.id, "failed_processing");
      throw procErr;
    }

    // Delete source upload before streaming processed files to R2
    try { await fs.unlink(uploadedPath); } catch {}

    await uploadFileToR2(R2_BUCKET_NAME, newClipKey, clipPath, "video/mp4");
    await uploadFileToR2(R2_BUCKET_NAME, newThumbKey, thumbPath, "image/jpeg");

    await updateSupabaseStatus(rec.id, "processed", { processed_at: new Date().toISOString() });

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
