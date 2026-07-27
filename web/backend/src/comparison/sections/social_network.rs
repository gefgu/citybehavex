//! `social-network` chart section: reads the simulation's social-network
//! sidecar JSON as-is (already shaped for the frontend graph renderer),
//! capping the visualized node/edge count for very large populations.
//! Mirrors `legacy Python backend/payload/legacy.py`'s social-network section.

use crate::payload::{ComparisonContext, empty_chart_payload};
use serde_json::{Value, json};

pub fn social_network_section_payload(ctx: &ComparisonContext) -> anyhow::Result<Value> {
    let mut payload = empty_chart_payload(ctx, Vec::new());
    let path = {
        let stem = ctx
            .synthetic_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        ctx.synthetic_path
            .with_file_name(format!("{stem}_social_network.json"))
    };
    if !path.exists() {
        return Ok(payload);
    }
    let mut data: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let nodes_len = data
        .get("nodes")
        .and_then(Value::as_array)
        .map_or(0usize, Vec::len);
    let edges_len = data
        .get("edges")
        .and_then(Value::as_array)
        .map_or(0usize, Vec::len);
    if data
        .get("node_count")
        .and_then(Value::as_u64)
        .unwrap_or(nodes_len as u64)
        != nodes_len as u64
        || data
            .get("edge_count")
            .and_then(Value::as_u64)
            .unwrap_or(edges_len as u64)
            != edges_len as u64
    {
        anyhow::bail!("social network sidecar count mismatch: {}", path.display());
    }
    const MAX_AGENTS: usize = 5000;
    if nodes_len > MAX_AGENTS {
        if let Some(obj) = data.as_object_mut() {
            let visible = MAX_AGENTS;
            if let Some(nodes) = obj.get_mut("nodes").and_then(Value::as_array_mut) {
                nodes.truncate(visible);
            }
            if let Some(edges) = obj.get_mut("edges").and_then(Value::as_array_mut) {
                edges.retain(|row| {
                    row.as_array().is_some_and(|r| {
                        r.len() >= 2
                            && r[0].as_u64().unwrap_or(u64::MAX) < visible as u64
                            && r[1].as_u64().unwrap_or(u64::MAX) < visible as u64
                    })
                });
            }
            if let Some(degrees) = obj.get_mut("degrees").and_then(Value::as_array_mut) {
                degrees.truncate(visible);
            }
            obj.insert("nodes_sampled".to_string(), json!(true));
            obj.insert("edges_sampled".to_string(), json!(true));
        }
    }
    payload["social_network"] = data;
    Ok(payload)
}
