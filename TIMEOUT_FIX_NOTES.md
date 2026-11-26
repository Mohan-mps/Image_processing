# Timeout Error Fix - Complete Solution

## Problem Summary
The application was experiencing `ReadTimeoutError` when connecting to the remote Ollama API at `aisandbox.mpslimited.com`. The errors occurred because:

1. **30-second timeout was too short** - The VLM model often needs more time to process chart images
2. **Only 2 retries** (3 total attempts) with minimal backoff wasn't sufficient for intermittent network issues
3. **No exponential backoff** - Quick retries could hit the same issue repeatedly

## Root Causes Identified

### 1. Network Latency
- Remote API hosted at `aisandbox.mpslimited.com`
- SSL/TLS handshake overhead
- Large image payloads being transmitted
- Model inference time on remote server

### 2. Insufficient Retry Strategy
- Only 2 retries with 0.5-1 second delays
- No exponential backoff for transient failures
- Timeout errors counted as final failures too quickly

## Solutions Implemented

### 1. Increased Default Timeout (30s → 180s)
**File: `src/main.py`**
```python
parser.add_argument("--ollama-timeout", type=float, default=180.0)
```
- Provides 6x more time for model inference
- Accounts for network latency and large payloads
- Can still be overridden via command line

### 2. Increased Default Retries (2 → 4)
**File: `src/main.py`**
```python
parser.add_argument("--ollama-retries", type=int, default=4)
```
- Now attempts up to 5 times total (1 initial + 4 retries)
- Significantly improves reliability for transient failures

### 3. Exponential Backoff Strategy
**File: `src/vlm_extractor.py`**
```python
sleep_time = self.retry_backoff * (2 ** attempt)
```
- Retry delays: 2s, 4s, 8s, 16s (with default backoff=2.0)
- Prevents overwhelming the server with rapid retries
- Gives transient issues time to resolve

### 4. Configurable Retry Backoff
**File: `src/main.py`**
```python
parser.add_argument("--ollama-retry-backoff", type=float, default=2.0)
```
- Allows fine-tuning backoff strategy
- Default 2.0 seconds provides good balance

## Updated Files
1. `src/vlm_extractor.py` - Core retry logic with exponential backoff
2. `src/main.py` - Increased default timeout and retries
3. `src/pipeline.py` - Applied new defaults
4. `src/evaluation.py` - Applied new defaults
5. `src/vlm_classifier.py` - Updated to support new parameters

## How to Use

### Option 1: Use New Defaults (Recommended)
Simply run your command as before - the new defaults will apply:
```bash
python -m src.main --input input
```

**New defaults:**
- Timeout: 180 seconds (was 30)
- Retries: 4 (was 2)
- Retry backoff: 2 seconds exponential

### Option 2: Custom Timeout Settings
Override the defaults for your specific needs:

```bash
# For very slow connections or complex charts
python -m src.main --input input --ollama-timeout 180.0 --ollama-retries 5

# For faster connections (but less reliable)
python -m src.main --input input --ollama-timeout 60.0 --ollama-retries 3

# Customize retry backoff
python -m src.main --input input --ollama-retry-backoff 3.0
```

### Option 3: Maximum Reliability
For unreliable networks or when you must complete processing:
```bash
python -m src.main --input input \
  --ollama-timeout 300.0 \
  --ollama-retries 6 \
  --ollama-retry-backoff 3.0
```

This provides:
- Up to 5 minutes per request
- 7 total attempts (1 + 6 retries)
- Longer backoff delays (3s, 6s, 12s, 24s, 48s, 96s)

## Expected Behavior

### Success Case
```
2026-11-24 11:04:54,110 INFO src.chart_detector: Detected 9 chart region(s)
2026-11-24 11:28:43,116 INFO chartxtract: Wrote output/ac6c03182__Im3.json with 9 chart(s)
```

### Retry Case (Now Recoverable)
```
2026-11-24 11:13:52,834 WARNING src.vlm_extractor: Ollama request failed (1/4): Read timed out. Retrying in 2.0 seconds...
2026-11-24 11:14:03,845 INFO chartxtract: Request succeeded on retry
```

### Still Failing After All Retries
If you still see errors after 5 attempts with the new settings:
1. Check your internet connection
2. Verify `aisandbox.mpslimited.com` is accessible
3. Contact your network administrator about firewall/proxy settings
4. Consider increasing timeout further: `--ollama-timeout 300.0`

## Technical Details

### Exponential Backoff Formula
```
sleep_time = retry_backoff * (2 ^ attempt_number)
```

With default `retry_backoff=2.0`:
- Attempt 0: No delay (initial request)
- Attempt 1: 2.0 seconds delay
- Attempt 2: 4.0 seconds delay
- Attempt 3: 8.0 seconds delay
- Attempt 4: 16.0 seconds delay

### Total Timeout Calculation
Maximum time per chart (worst case):
```
max_time = (timeout * (retries + 1)) + sum(backoff delays)
         = (180 * 5) + (2 + 4 + 8 + 16)
         = 900 + 30
         = 930 seconds (~15.5 minutes)
```

## Monitoring Recommendations

1. **Watch for persistent timeouts** - If the same chart consistently fails, the issue may be with the chart complexity, not the network

2. **Check retry patterns** - If retries consistently succeed on attempt 2-3, your network has transient issues

3. **Monitor total processing time** - If average time per chart increases significantly, consider:
   - Load on the remote server
   - Your network bandwidth
   - Chart complexity

## Troubleshooting

### Still getting timeouts?
```bash
# Try even more aggressive settings
python -m src.main --input input \
  --ollama-timeout 300.0 \
  --ollama-retries 8 \
  --ollama-retry-backoff 5.0
```

### Processing too slow?
```bash
# Reduce retries if you have a stable connection
python -m src.main --input input \
  --ollama-timeout 90.0 \
  --ollama-retries 2 \
  --ollama-retry-backoff 1.0
```

### Need to debug?
```bash
# Enable debug logging to see detailed retry behavior
python -m src.main --input input --log-level DEBUG
```

## Verification

The fix is working correctly if you see:
1. ✅ No more immediate timeout failures
2. ✅ Warning messages showing retry attempts with delays
3. ✅ Successful completion after retries
4. ✅ All charts processed without crashes

## Summary

This comprehensive fix addresses the timeout issues by:
- **6x longer timeout** (180s vs 30s)
- **2x more retries** (4 vs 2)
- **Smart exponential backoff** (2s, 4s, 8s, 16s)
- **Full configurability** for different network conditions

The error should now be fully arrested and the application should reliably process all charts even with intermittent network issues.
