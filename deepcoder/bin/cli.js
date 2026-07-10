#!/usr/bin/env node

/**
 * deepcoder — token-efficient CLI coding assistant (NVIDIA DeepSeek).
 *
 * Usage:
 *   deepcoder              Start interactive REPL
 *   deepcoder --prompt "..."   Run a single prompt and exit
 *   deepcoder --help       Show help
 *   deepcoder --model "..."    Use a specific DeepSeek model
 *   deepcoder --budget 50000   Set token budget
 *   deepcoder --verbose    Show LLM responses
 */

import { CLI } from '../src/cli.js';
import { Agent } from '../src/agent.js';

const args = process.argv.slice(2);

// ── Help ──────────────────────────────────────────────────────────
if (args.includes('--help') || args.includes('-h')) {
  console.log(`
🧠 deepcoder — AI coding assistant (NVIDIA DeepSeek)

A token-efficient CLI tool that helps you code without burning through your API budget.

Usage:
  deepcoder                       Start interactive REPL
  deepcoder --prompt "..."        Run a single prompt, then exit
  deepcoder --verbose             Verbose mode (shows LLM raw responses)
  deepcoder --model deepseek-ai/deepseek-v3   Use a specific model
  deepcoder --budget 50000        Limit session to 50K tokens (default: 100K)
  deepcoder --help                Show this help

Environment variables:
  NVIDIA_API_KEY    Your NVIDIA DeepSeek API key (required if not prompted)

Models (NVIDIA NIM catalog):
  deepseek-ai/deepseek-v4-flash   Default — fast, token-efficient
  deepseek-ai/deepseek-v4-pro     More capable (uses more tokens)
  deepseek-ai/deepseek-coder-6.7b-instruct  Lighter coding model

Tips:
  - Start in your project directory
  - The tool reads files, makes edits, runs commands
  - Use /budget to check token usage mid-session
  - Use /model to switch models without restarting
`);
  process.exit(0);
}

// ── Parse options ─────────────────────────────────────────────────
const options = {};

options.verbose = args.includes('--verbose');

// Budget
const budgetIndex = args.indexOf('--budget');
if (budgetIndex !== -1 && args.length > budgetIndex + 1) {
  const n = parseInt(args[budgetIndex + 1], 10);
  if (!isNaN(n) && n >= 1000) {
    options.sessionBudget = n;
  }
}

// Model
const modelIndex = args.indexOf('--model');
if (modelIndex !== -1 && args.length > modelIndex + 1) {
  options.model = args[modelIndex + 1];
}

// Single prompt mode
const promptIndex = args.indexOf('--prompt');
let singlePrompt = null;
if (promptIndex !== -1 && args.length > promptIndex + 1) {
  singlePrompt = args[promptIndex + 1];
}

// ── Run ───────────────────────────────────────────────────────────
if (singlePrompt) {
  // Single-prompt mode — requires API key as env var
  if (!process.env.NVIDIA_API_KEY && !process.env.DEEPSEEK_API_KEY) {
    console.error('\n❌ Single-prompt mode requires NVIDIA_API_KEY environment variable.\n');
    process.exit(1);
  }
  console.log('\n🧠 Processing...\n');
  try {
    // Initialize agent directly (not via CLI.start())
    const agent = new Agent(options);
    const response = await agent.processUserInput(singlePrompt);
    console.log(response);
  } catch (err) {
    console.error('Error:', err.message);
    process.exit(1);
  }
  process.exit(0);
} else {
  // Interactive REPL (CLI.start() initializes the agent)
  const cli = new CLI(options);
  await cli.start();
}
