//! MOSS-Image layout prompts: place text, objects and characters with boxes.
//!
//! MOSS-Image reads a JSON layout as its prompt: a `summary` of the whole image
//! plus `items`, each with a label, a caption and a box in pixel coordinates
//! relative to the canvas center (`abs_centralized`, `yxyx`). Writing those
//! centered coordinates by hand (or asking an LLM to) is error-prone, so the
//! upstream prompt-enhancer skill defines a readable *draft*: ordinary
//! top-left pixel boxes (`abs`, `xyxy`) plus the canvas `width`/`height`, and a
//! compiler that turns it into the compact model-facing prompt.
//!
//! This is a port of that compiler (`skills/moss-image-prompt-enhancer/scripts/
//! compile_prompt.py`, OpenMOSS/MOSS-Image, Apache-2.0) with the same rules:
//! unknown fields are rejected rather than dropped, boxes are enclosed in pixel
//! edges (floor left/top, ceil right/bottom) before centering, and key order is
//! kept because the model was trained on the compiler's output.
//!
//! Three inputs are accepted:
//! - a draft (`width`, `height`, `bbox_coord_space: "abs"`, `bbox_format: "xyxy"`),
//! - an already-compiled layout (`abs_centralized` / `yxyx`), passed through,
//! - a caption draft `{"caption": "..."}`, which compiles to the plain caption.

use anyhow::{Result, bail};
use indexmap::IndexMap;
use serde::de::{Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::Value;
use std::fmt;

/// A JSON object that rejects duplicate keys and keeps key order.
#[derive(Debug, Clone, PartialEq)]
struct Obj(IndexMap<String, Value>);

impl<'de> Deserialize<'de> for Obj {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Obj;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Obj, A::Error> {
                let mut out = IndexMap::new();
                while let Some((k, v)) = map.next_entry::<String, Value>()? {
                    if out.contains_key(&k) {
                        return Err(serde::de::Error::custom(format!("Duplicate JSON key: {k}")));
                    }
                    out.insert(k, v);
                }
                Ok(Obj(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// Parse layout JSON text: rejects duplicate keys, NaN/Infinity (not valid JSON).
fn parse(text: &str) -> Result<Obj> {
    let root: Obj =
        serde_json::from_str(text).map_err(|e| anyhow::anyhow!("Invalid layout JSON: {e}"))?;
    Ok(root)
}

/// `summary` and `items` re-read with `Obj` so their key order is kept.
#[derive(serde::Deserialize)]
struct RawDoc {
    #[serde(default)]
    summary: Option<Obj>,
    #[serde(default)]
    items: Option<Vec<Obj>>,
}

fn require(cond: bool, msg: impl Into<String>) -> Result<()> {
    if !cond {
        bail!("{}", msg.into());
    }
    Ok(())
}

fn check_fields(
    obj: &IndexMap<String, Value>,
    allowed: &[&str],
    required: &[&str],
    loc: &str,
) -> Result<()> {
    let mut unknown: Vec<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|k| !allowed.contains(k))
        .collect();
    unknown.sort_unstable();
    require(
        unknown.is_empty(),
        format!("{loc}: unknown fields {unknown:?}"),
    )?;
    let mut missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|k| !obj.contains_key(*k))
        .collect();
    missing.sort_unstable();
    require(
        missing.is_empty(),
        format!("{loc}: missing fields {missing:?}"),
    )
}

fn nonempty_text(v: &Value, loc: &str) -> Result<()> {
    require(
        v.as_str().is_some_and(|s| !s.trim().is_empty()),
        format!("{loc} must be nonempty text"),
    )
}

fn strings(v: &Value, loc: &str) -> Result<()> {
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("{loc} must be an array"))?;
    for (i, item) in arr.iter().enumerate() {
        nonempty_text(item, &format!("{loc}[{i}]"))?;
    }
    Ok(())
}

/// Python's `round(x, 9)` closely enough for edge enclosure.
fn round9(x: f64) -> f64 {
    (x * 1e9).round() / 1e9
}

/// Top-left `xyxy` pixel box -> centered `yxyx` box.
pub fn centered_box(b: &Value, width: i64, height: i64, loc: &str) -> Result<[i64; 4]> {
    let arr = b.as_array().filter(|a| a.len() == 4);
    let arr = arr.ok_or_else(|| anyhow::anyhow!("{loc} must be [x1,y1,x2,y2]"))?;
    let mut v = [0f64; 4];
    for (i, n) in arr.iter().enumerate() {
        let x = n.as_f64().filter(|x| x.is_finite() && n.is_number());
        v[i] = x.ok_or_else(|| anyhow::anyhow!("{loc} must contain four finite numbers"))?;
    }
    let [x1, y1, x2, y2] = v;
    require(
        x1 < x2 && y1 < y2,
        format!("{loc} must have positive width and height"),
    )?;
    require(
        0.0 <= x1 && x2 <= width as f64 && 0.0 <= y1 && y2 <= height as f64,
        format!("{loc} exceeds the {width}x{height} canvas"),
    )?;
    let r = [
        round9(y1).floor() as i64 - height / 2,
        round9(x1).floor() as i64 - width / 2,
        round9(y2).ceil() as i64 - height / 2,
        round9(x2).ceil() as i64 - width / 2,
    ];
    require(
        r[0] < r[2] && r[1] < r[3],
        format!("{loc} is too small at export precision"),
    )?;
    Ok(r)
}

/// The result of compiling a layout: the prompt string and, for drafts, the
/// canvas size the boxes were drawn on (generation must use it).
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledLayout {
    pub prompt: String,
    pub size: Option<(u32, u32)>,
}

fn validate_summary(summary: &IndexMap<String, Value>) -> Result<()> {
    check_fields(
        summary,
        &["caption", "meta", "canvas", "palette"],
        &["caption"],
        "summary",
    )?;
    nonempty_text(&summary["caption"], "summary.caption")?;
    if let Some(c) = summary.get("canvas") {
        require(
            matches!(c.as_str(), Some("single" | "multi_panel" | "collage")),
            "summary.canvas must be single, multi_panel or collage",
        )?;
    }
    for key in ["meta", "palette"] {
        if let Some(v) = summary.get(key) {
            strings(v, &format!("summary.{key}"))?;
        }
    }
    Ok(())
}

fn validate_item(item: &IndexMap<String, Value>, loc: &str) -> Result<()> {
    check_fields(
        item,
        &["label", "caption", "bbox", "text", "tags", "palette"],
        &["label", "caption", "bbox"],
        loc,
    )?;
    require(
        matches!(
            item["label"].as_str(),
            Some("object" | "character" | "text")
        ),
        format!("{loc}.label must be object, character or text"),
    )?;
    nonempty_text(&item["caption"], &format!("{loc}.caption"))?;
    if item["label"].as_str() == Some("text") {
        require(
            item.contains_key("text"),
            format!("{loc}: text items require literal text"),
        )?;
    }
    if let Some(t) = item.get("text") {
        nonempty_text(t, &format!("{loc}.text"))?;
    }
    for key in ["tags", "palette"] {
        if let Some(v) = item.get(key) {
            strings(v, &format!("{loc}.{key}"))?;
        }
    }
    Ok(())
}

fn positive_multiple_of_16(v: &Value, name: &str) -> Result<i64> {
    match v.as_u64() {
        Some(n) if v.is_u64() && n > 0 && n % 16 == 0 => Ok(n as i64),
        _ => bail!("{name} must be a positive integer multiple of 16"),
    }
}

/// Compile a layout given as JSON text.
pub fn compile(text: &str) -> Result<CompiledLayout> {
    let root = parse(text)?.0;
    let raw: RawDoc =
        serde_json::from_str(text).map_err(|e| anyhow::anyhow!("Invalid layout JSON: {e}"))?;

    if root.contains_key("caption") {
        check_fields(&root, &["caption"], &["caption"], "Caption draft")?;
        nonempty_text(&root["caption"], "caption")?;
        return Ok(CompiledLayout {
            prompt: root["caption"].as_str().unwrap().to_string(),
            size: None,
        });
    }

    let space = root.get("bbox_coord_space").and_then(Value::as_str);
    let summary = raw.summary.map(|o| o.0);
    let items = raw
        .items
        .map(|v| v.into_iter().map(|o| o.0).collect::<Vec<_>>());

    if space == Some("abs_centralized") {
        // Already compiled: validate the shape, then pass through compactly.
        check_fields(
            &root,
            &["summary", "items", "bbox_coord_space", "bbox_format"],
            &["summary", "items", "bbox_coord_space", "bbox_format"],
            "Layout",
        )?;
        require(
            root["bbox_format"].as_str() == Some("yxyx"),
            "Compiled layout bbox_format must be yxyx",
        )?;
        let summary = summary.ok_or_else(|| anyhow::anyhow!("summary must be an object"))?;
        validate_summary(&summary)?;
        let items = items.filter(|i| !i.is_empty());
        let items = items.ok_or_else(|| anyhow::anyhow!("items must be a nonempty flat array"))?;
        for (i, item) in items.iter().enumerate() {
            validate_item(item, &format!("items[{i}]"))?;
        }
        return Ok(CompiledLayout {
            prompt: compact_ordered(&summary, &items)?,
            size: None,
        });
    }

    let root_fields = [
        "width",
        "height",
        "bbox_coord_space",
        "bbox_format",
        "summary",
        "items",
    ];
    check_fields(&root, &root_fields, &root_fields, "Layout draft")?;
    let width = positive_multiple_of_16(&root["width"], "width")?;
    let height = positive_multiple_of_16(&root["height"], "height")?;
    require(space == Some("abs"), "Draft bbox_coord_space must be abs")?;
    require(
        root["bbox_format"].as_str() == Some("xyxy"),
        "Draft bbox_format must be xyxy",
    )?;

    let summary = summary.ok_or_else(|| anyhow::anyhow!("summary must be an object"))?;
    validate_summary(&summary)?;
    let items = items.filter(|i| !i.is_empty());
    let items = items.ok_or_else(|| anyhow::anyhow!("items must be a nonempty flat array"))?;

    let mut compiled = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let loc = format!("items[{i}]");
        validate_item(item, &loc)?;
        let mut out = IndexMap::new();
        for (k, v) in item {
            if k == "bbox" {
                let b = centered_box(v, width, height, &format!("{loc}.bbox"))?;
                out.insert(k.clone(), serde_json::json!(b));
            } else {
                out.insert(k.clone(), v.clone());
            }
        }
        compiled.push(out);
    }
    Ok(CompiledLayout {
        prompt: compact_ordered(&summary, &compiled)?,
        size: Some((width as u32, height as u32)),
    })
}

/// Serialize with every object in insertion order.
fn compact_ordered(
    summary: &IndexMap<String, Value>,
    items: &[IndexMap<String, Value>],
) -> Result<String> {
    #[derive(serde::Serialize)]
    struct Out<'a> {
        summary: &'a IndexMap<String, Value>,
        items: &'a [IndexMap<String, Value>],
        bbox_coord_space: &'static str,
        bbox_format: &'static str,
    }
    Ok(serde_json::to_string(&Out {
        summary,
        items,
        bbox_coord_space: "abs_centralized",
        bbox_format: "yxyx",
    })?)
}

/// True for model ids whose prompts understand MOSS layouts.
pub fn model_supports_layout(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    id.starts_with("moss-image") || id.starts_with("kromoss")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn draft() -> Value {
        json!({
            "width": 1536,
            "height": 1024,
            "bbox_coord_space": "abs",
            "bbox_format": "xyxy",
            "summary": {"meta": ["poster"], "canvas": "single", "caption": "A ceramic campaign."},
            "items": [{"label": "text", "caption": "Dark-green lettering.",
                       "text": "把日常，过成诗。\n\"MOSS\" 🪴", "bbox": [928, 272, 1408, 448]}]
        })
    }

    fn compile_value(v: &Value) -> Result<CompiledLayout> {
        compile(&serde_json::to_string(v).unwrap())
    }

    #[test]
    fn landscape_conversion_preserves_content() {
        let out = compile_value(&draft()).unwrap();
        assert_eq!(out.size, Some((1536, 1024)));
        let r: Value = serde_json::from_str(&out.prompt).unwrap();
        assert_eq!(r["items"][0]["bbox"], json!([-240, 160, -64, 640]));
        assert_eq!(r["items"][0]["text"], draft()["items"][0]["text"]);
        assert_eq!(r["summary"], draft()["summary"]);
        assert_eq!(r["bbox_coord_space"], "abs_centralized");
        assert_eq!(r["bbox_format"], "yxyx");
        assert!(r.get("width").is_none());
        assert!(!out.prompt.contains('\n'));
    }

    #[test]
    fn output_is_byte_identical_to_upstream_compiler() {
        // Expected string produced by upstream compile_prompt.py on this draft.
        let draft = r#"{"width": 1536, "height": 1024, "bbox_coord_space": "abs", "bbox_format": "xyxy",
 "summary": {"meta": ["poster"], "canvas": "single", "caption": "A ceramic campaign."},
 "items": [{"label": "text", "caption": "Dark-green lettering.", "text": "把日常，过成诗。\n\"MOSS\" 🪴", "bbox": [928, 272, 1408, 448]},
           {"label": "object", "caption": "A cup.", "bbox": [10.9, 20.1, 100.01, 200.99], "tags": ["ceramic"]}]}"#;
        let expected = r#"{"summary":{"meta":["poster"],"canvas":"single","caption":"A ceramic campaign."},"items":[{"label":"text","caption":"Dark-green lettering.","text":"把日常，过成诗。\n\"MOSS\" 🪴","bbox":[-240,160,-64,640]},{"label":"object","caption":"A cup.","bbox":[-492,-758,-311,-667],"tags":["ceramic"]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}"#;
        assert_eq!(compile(draft).unwrap().prompt, expected);
    }

    #[test]
    fn portrait_edges_and_fractional_enclosure() {
        let b = |v: Value| centered_box(&v, 768, 1344, "box").unwrap();
        assert_eq!(b(json!([0, 0, 768, 1344])), [-672, -384, 672, 384]);
        assert_eq!(
            b(json!([10.9, 20.1, 100.01, 200.99])),
            [-652, -374, -471, -283]
        );
        assert_eq!(
            b(json!([1.00000000001, 2, 100, 200])),
            [-670, -383, -472, -284]
        );
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        for bad in [
            json!([-1, 0, 100, 100]),
            json!([0, 0, 1537, 100]),
            json!([0, 0, 100, 1025]),
            json!([100, 0, 50, 100]),
            json!([0, 0, 0, 100]),
            json!([0, 0, 100]),
            json!([true, 0, 100, 100]),
            json!([0, 0, "100", 100]),
            json!([0, 0, 0.00000000001, 100]),
        ] {
            assert!(centered_box(&bad, 1536, 1024, "box").is_err(), "{bad}");
        }
    }

    #[test]
    fn invalid_dimensions_and_conventions_are_rejected() {
        for (field, value) in [
            ("width", json!(1023)),
            ("height", json!(0)),
            ("width", json!(true)),
            ("height", json!(1024.0)),
            ("bbox_coord_space", json!("normalized_1000")),
            ("bbox_format", json!("yxyx")),
        ] {
            let mut d = draft();
            d[field] = value.clone();
            assert!(compile_value(&d).is_err(), "{field}={value}");
        }
    }

    #[test]
    fn unrepresentable_content_is_not_silently_dropped() {
        for (field, value) in [
            ("children", json!([{"bbox": [0, 0, 20, 20]}])),
            ("font_size", json!(32)),
            ("parent", json!("cup")),
        ] {
            let mut d = draft();
            d["items"][0][field] = value;
            assert!(compile_value(&d).is_err(), "{field}");
        }
        let mut d = draft();
        d["items"][0].as_object_mut().unwrap().remove("text");
        let err = compile_value(&d).unwrap_err().to_string();
        assert!(err.contains("literal text"), "{err}");
    }

    #[test]
    fn caption_is_plain_text() {
        let caption = "An ivory cup marked \"茶\".\nPreserve this second line.";
        assert_eq!(
            compile_value(&json!({"caption": caption})).unwrap().prompt,
            caption
        );
        assert!(compile_value(&json!({"caption": ""})).is_err());
        assert!(compile_value(&json!({"caption": caption, "items": []})).is_err());
    }

    #[test]
    fn duplicate_keys_and_nonfinite_constants_are_rejected() {
        assert!(compile(r#"{"caption":"one","caption":"two"}"#).is_err());
        assert!(compile(r#"{"caption":NaN}"#).is_err());
        assert!(compile(r#"{"caption":Infinity}"#).is_err());
    }

    #[test]
    fn compiled_layout_passes_through() {
        let compiled = compile_value(&draft()).unwrap().prompt;
        let again = compile(&compiled).unwrap();
        assert_eq!(again.prompt, compiled);
        assert_eq!(again.size, None);
    }

    #[test]
    fn layout_models() {
        assert!(model_supports_layout("moss-image"));
        assert!(model_supports_layout("moss-image-base"));
        assert!(model_supports_layout("kromoss"));
        assert!(!model_supports_layout("flux2-klein-9b"));
    }
}
