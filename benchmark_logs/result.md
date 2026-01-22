============================================================
🚀 MECE Framework vs LangGraph Benchmark
============================================================

🔌 Using Real LLM API:
   Base URL: https://requests-floating-toll-travesti.trycloudflare.com/v1
   Model: Qwen3-VL

============================================================
📝 Test: Single Agent (Math)
   Query: Calculate 25 * 17 and explain your reasoning step ...
   Expected Answer: 425
============================================================

  Run 1/1:
  ✅ MECE
     Latency: 7795.37 ms
     Tokens: 539
     Agents: 1
     Answer: ✓ Correct answer found in response
  ✅ LangGraph
     Latency: 13026.43 ms
     Tokens: 895
     Agents: 1
     Answer: ✓ Correct answer found in response

--- Summary: Single Agent (Math) ---

  📊 MECE
     Runs: 1
     Avg Latency: 7795.37 ms
     Min/Max: 7795.37 / 7795.37 ms
     Std Dev: 0.00 ms
     Avg Tokens: 539
     Success Rate: 100%

  📊 LangGraph
     Runs: 1
     Avg Latency: 13026.43 ms
     Min/Max: 13026.43 / 13026.43 ms
     Std Dev: 0.00 ms
     Avg Tokens: 895
     Success Rate: 100%

  ⚖️  Comparison:
     MECE faster by 5231.06 ms (40.2%)
     MECE uses 356 fewer tokens

============================================================
📝 Test: Three Agents (Chain)
   Query: Analyze the mathematical properties of prime numbe...
============================================================

  Run 1/1:
  ✅ MECE
     Latency: 47771.87 ms
     Tokens: 5292
     Agents: 3
     Answer: ✓ All keywords found
  ✅ LangGraph
     Latency: 47925.80 ms
     Tokens: 6293
     Agents: 3
     Answer: ✓ All keywords found

--- Summary: Three Agents (Chain) ---

  📊 MECE
     Runs: 1
     Avg Latency: 47771.87 ms
     Min/Max: 47771.87 / 47771.87 ms
     Std Dev: 0.00 ms
     Avg Tokens: 5292
     Success Rate: 100%

  📊 LangGraph
     Runs: 1
     Avg Latency: 47925.80 ms
     Min/Max: 47925.80 / 47925.80 ms
     Std Dev: 0.00 ms
     Avg Tokens: 6293
     Success Rate: 100%

  ⚖️  Comparison:
     MECE faster by 153.93 ms (0.3%)
     MECE uses 1001 fewer tokens

============================================================
📝 Test: Parallel Agents
   Query: Analyze the mathematical properties of prime numbe...
============================================================

  Run 1/1:
  ✅ MECE
     Latency: 47754.04 ms
     Tokens: 5289
     Agents: 3
     Answer: ✓ All keywords found
  ✅ LangGraph
     Latency: 48126.36 ms
     Tokens: 6301
     Agents: 3
     Answer: ✓ All keywords found

--- Summary: Parallel Agents ---

  📊 MECE
     Runs: 1
     Avg Latency: 47754.04 ms
     Min/Max: 47754.04 / 47754.04 ms
     Std Dev: 0.00 ms
     Avg Tokens: 5289
     Success Rate: 100%

  📊 LangGraph
     Runs: 1
     Avg Latency: 48126.36 ms
     Min/Max: 48126.36 / 48126.36 ms
     Std Dev: 0.00 ms
     Avg Tokens: 6301
     Success Rate: 100%

  ⚖️  Comparison:
     MECE faster by 372.32 ms (0.8%)
     MECE uses 1012 fewer tokens

============================================================
📡 Streaming Benchmarks
============================================================

📝 Test: Single Agent Streaming
  ✅ MECE (Streaming)
     Latency: 12180.30 ms
     Tokens: 240
     Agents: 1
  ✅ LangGraph (Streaming)
     Latency: 10139.15 ms
     Tokens: 704
     Agents: 1

📝 Test: Three Agents Streaming
  ✅ MECE (Streaming)
     Latency: 47947.98 ms
     Tokens: 2632
     Agents: 3
  ✅ LangGraph (Streaming)
     Latency: 48007.24 ms
     Tokens: 6294
     Agents: 3

============================================================
📈 Final Summary
============================================================

MECE Framework:
  Total runs: 3
  Average latency: 34440.43 ms

LangGraph:
  Total runs: 3
  Average latency: 36359.53 ms

🏆 MECE Framework is 1919.11 ms faster on average!

--- Answer Accuracy ---
  MECE_Single Agent (Math): 1/1 (100.0%)
  LangGraph_Single Agent (Math): 1/1 (100.0%)
  MECE_Three Agents (Chain): 1/1 (100.0%)
  LangGraph_Three Agents (Chain): 1/1 (100.0%)
  MECE_Parallel Agents: 1/1 (100.0%)
  LangGraph_Parallel Agents: 1/1 (100.0%)

📁 Conversation logs saved to: benchmark_logs/benchmark_20251210_024655.json

============================================================
✅ Benchmark completed!
============================================================

============================================================
📊 MECE Framework Metrics Demo
============================================================

Running 5 iterations to collect metrics...
  Run 1: 110 tokens, 12883.93 ms
  Run 2: 129 tokens, 9262.35 ms
  Run 3: 50 tokens, 8830.51 ms
  Run 4: 150 tokens, 15020.41 ms
  Run 5: 113 tokens, 9902.39 ms

--- Node Metrics ---

  solver:
    Executions: 5
    Reliability: 100.00%
    Avg Latency: 5589.96 ms
    Avg Tokens: 55
    Quality: 0.90

  checker:
    Executions: 5
    Reliability: 100.00%
    Avg Latency: 5589.96 ms
    Avg Tokens: 55
    Quality: 0.90

--- Edge Metrics ---

  solver -> checker:
    Transitions: 5
    Reliability: 100.00%
    Avg Latency: 10.00 ms

--- Routing Weights ---
  solver -> checker: 0.2300

--- Node Scores ---
  solver: 0.7690
  checker: 0.7690

--- Global Metrics ---
  Avg Latency: 5887.63 ms
  Avg Tokens: 55
  Avg Quality: 0.90
