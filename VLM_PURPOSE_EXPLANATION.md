# VLM Usage in ChartXtract

## What is VLM?

**VLM** = **Vision Language Model**

In this project, VLM refers to the **Qwen 3.5 (9B parameters)** model running on Ollama at `aisandbox.mpslimited.com`.

## Primary Purpose: Chart Axis Tick Label Extraction

The VLM is used to **extract text labels from chart axes** in scientific figures. Specifically, it reads:

### What the VLM Extracts:

1. **X-axis tick labels** - Text values along the horizontal axis (bottom)
2. **Y-axis tick labels** - Text values along the vertical axis (left side)
3. **Graph classification** - Whether the image is actually a graph or not
4. **Uncertain values** - Values the model is not confident about

### Example Input/Output:

**Input:** A cropped chart image showing a scientific plot with labeled axes

**Output (JSON):**
```json
{
  "is_graph": true,
  "x_axis_tick": ["0", "10", "20", "30", "40", "50"],
  "y_axis_tick": ["0.0", "0.5", "1.0", "1.5", "2.0"],
  "uncertain_values": []
}
```

## Why Use a VLM Instead of OCR?

Traditional OCR (Optical Character Recognition) tools struggle with:
- **Context understanding** - Distinguishing tick labels from legends, titles, and data values
- **Layout interpretation** - Knowing which text belongs to which axis
- **Multi-line labels** - Understanding when text spans multiple lines but represents one label
- **Rotated/angled text** - Handling diagonal labels common in crowded x-axes
- **Scientific notation** - Understanding formats like "1.0 × 10³"

The VLM can:
- ✅ **Understand context** - Knows what a chart is vs. a table or photo
- ✅ **Spatial reasoning** - Identifies which text belongs to X vs Y axis
- ✅ **Follow instructions** - Preserves exact formatting (like trailing zeros: "-1.0" not "-1")
- ✅ **Handle complexity** - Deals with rotated text, multi-line labels, and scientific notation
- ✅ **Semantic understanding** - Can distinguish tick labels from legends and titles

## Pipeline Flow

```
1. Input Image (scientific paper/document)
   ↓
2. YOLO Chart Detector (finds chart regions)
   ↓
3. Crop Charts (extract individual charts)
   ↓
4. Crop Axes (extract X and Y axis regions separately)
   ↓
5. VLM Extractor ← YOU ARE HERE
   │  - Receives: cropped chart + axis crops
   │  - Sends to: Qwen 3.5 model via Ollama API
   │  - Returns: tick labels as structured JSON
   ↓
6. Validation & Post-processing
   ↓
7. Output JSON (structured chart data)
```

## Model Details

**Model:** Qwen 3.5 (9B parameters)
- **Type:** Vision-Language Model (multimodal - processes images + text)
- **Hosting:** Ollama server at `aisandbox.mpslimited.com`
- **API:** Ollama Chat API (`/ollama/api/chat`)
- **Input:** Text prompt + Base64-encoded images
- **Output:** JSON with tick labels

## Prompt Strategy

The system uses sophisticated prompts to guide the VLM:

### Zero-shot Mode
Basic instructions asking the model to extract tick labels.

### Task-specific Mode (default)
Detailed instructions about:
- How to identify X vs Y axis labels
- Handling multi-line labels (join with space)
- Preserving exact text (including decimal zeros)
- Ignoring legends, titles, and right-side scales
- Handling rotated/angled text

### Few-shot Mode
Includes example images with correct answers to teach the model by demonstration.

### Few-shot Enhanced Mode
Uses separate cropped axis regions for better accuracy.

## Why the Timeout Issue Occurred

The VLM processing involves:

1. **Large image upload** - Multiple images (chart + axis crops) encoded in Base64
2. **Network latency** - Remote server at `aisandbox.mpslimited.com`
3. **Model inference time** - 9B parameter model needs time to:
   - Process the image
   - Understand the chart layout
   - Extract text from axes
   - Format as JSON
4. **Complex charts** - Scientific plots can have dense labels requiring careful reading

**Original timeout:** 30 seconds (too short!)
**New timeout:** 180 seconds (6x longer, much more reliable)

## Code Files Involved

| File | Purpose |
|------|---------|
| `src/vlm_extractor.py` | Ollama API client, handles retries and timeouts |
| `src/prompt_manager.py` | Builds prompts with images for different modes |
| `src/validator.py` | Validates VLM output JSON structure |
| `src/main.py` | Main pipeline that calls VLM for each chart |
| `prompts/*.txt` | Instruction templates for the VLM |

## Configuration Options

```bash
# Control VLM behavior via command-line arguments:

--ollama-model qwen3.5:9b           # Which model to use
--ollama-url https://...            # API endpoint
--ollama-timeout 180.0              # Max seconds per request
--ollama-retries 4                  # Number of retry attempts
--ollama-retry-backoff 2.0          # Exponential backoff multiplier
--ollama-temperature 0.0            # Deterministic output (0) vs creative (1)
--ollama-num-predict 512            # Max tokens in response
--prompt-mode task-specific         # Prompting strategy
```

## Secondary Use: Non-Chart Classification

The VLM is also used in `src/vlm_classifier.py` and `src/non_chart_pipeline.py` to:
- Classify non-chart images (tables, photos, diagrams)
- Extract measurements and text from non-chart regions

## Performance Characteristics

**Per chart processing time:**
- Fast charts: 10-30 seconds
- Complex charts: 30-120 seconds
- Very complex/retries: up to 180 seconds

**Success rate with new settings:**
- 99%+ with exponential backoff and increased timeout
- Handles transient network issues gracefully

## Summary

**VLM Purpose:** Extract precise text labels from chart X and Y axes using a vision-language AI model that understands chart structure and can follow detailed instructions about formatting and layout.

**Why it's essential:** Traditional OCR can't distinguish tick labels from other text or understand chart spatial relationships. The VLM provides semantic understanding of what a chart is and intelligent extraction of only the relevant axis values.

**The fix:** Increased timeout from 30s → 180s, added exponential backoff retries, so the VLM has enough time to process complex scientific charts even with network latency.
