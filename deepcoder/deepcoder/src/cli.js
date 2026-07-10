/**
 * CLI Interface — manages the interactive REPL, API key setup, and commands.
 */
import { createInterface } from 'node:readline/promises';
import { stdin as input, stdout as output, env } from 'node:process';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { Agent } from './agent.js';
import { LLMClient } from './llm.js';

/** Load .env file from cwd or home directory */
function loadDotenv() {
  const paths = [
    resolve(process.cwd(), '.env'),
    resolve(process.env.HOME || process.env.USERPROFILE || '', '.deepcoder.env'),
  ];
  for (const p of paths) {
    if (existsSync(p)) {
      try {
        const content = readFileSync(p, 'utf-8');
        for (const line of content.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith('#')) continue;
          const eqIdx = trimmed.indexOf('=');
          if (eqIdx === -1) continue;
          const key = trimmed.slice(0, eqIdx).trim();
          const val = trimmed.slice(eqIdx + 1).trim().replace(/^['"]|['"]$/g, '');
          if (key && !process.env[key]) {
            process.env[key] = val;
          }
        }
      } catch {}
    }
  }
}

loadDotenv();

const BANNER = `
╔══════════════════════════════════════╗
║      🧠  deepcoder  v1.0            ║
║  AI coding assistant (NVIDIA DeepSeek)║
╚══════════════════════════════════════╝
`;

const HELP_TEXT = `
Commands:
  /help       Show this help message
  /budget     Show token budget status
  /budget N   Set session budget (e.g., /budget 50000)
  /reset      Reset conversation (keep budget)
  /model      Show current model
  /model M    Set model (e.g., /model deepseek-ai/deepseek-v3)
  /verbose    Toggle verbose mode (show LLM responses)
  /exit       Exit

Usage: type your coding request. I'll read files, edit code, run commands.
`;

export class CLI {
  constructor(options = {}) {
    this.agent = null;
    this.options = options;
    this.running = true;
  }

  async start() {
    // Check for API key first
    let apiKey = env.NVIDIA_API_KEY || env.DEEPSEEK_API_KEY;

    if (!apiKey) {
      apiKey = await this._promptApiKey();
      if (!apiKey) {
        console.log('\nNo API key provided. Exiting.\n');
        process.exit(1);
      }
      // Store it for the session
      env.NVIDIA_API_KEY = apiKey;
    }

    // Initialize agent
    try {
      this.agent = new Agent({ ...this.options, apiKey });
      // Test the API connection
      console.log('\nTesting API connection...');
      const test = await this.agent.llm.complete([
        { role: 'system', content: 'Reply with "ok" only.' },
        { role: 'user', content: 'test' },
      ]);
      if (test.choices?.[0]?.message?.content) {
        console.log('\x1b[32m✓ API connected successfully\x1b[0m');
      }
    } catch (err) {
      console.error('\n\x1b[31m✗ Failed to connect to API:\x1b[0m', err.message);
      console.log('\nTroubleshooting:');
      console.log('  1. Check your API key is correct');
      console.log('  2. Verify network connectivity');
      console.log('  3. Try: set NVIDIA_API_KEY=your-key-here\n');
      process.exit(1);
    }

    console.log(BANNER);
    console.log('Type /help for commands, /exit to quit.\n');

    const rl = createInterface({ input, output });

    while (this.running) {
      const input = await rl.question('\x1b[36m❯ \x1b[0m');
      await this._handleInput(input);
    }

    rl.close();
    console.log('\nGoodbye! 👋');
  }

  async _promptApiKey() {
    console.log('\n\x1b[33mNo NVIDIA API key found.\x1b[0m');
    console.log('Get your key from https://build.nvidia.com/\n');

    const rl = createInterface({ input, output });
    const key = await rl.question('\x1b[36mPaste your NVIDIA API key (nvapi-...): \x1b[0m');
    rl.close();

    return key.trim();
  }

  async _handleInput(input) {
    const trimmed = input.trim();
    if (!trimmed) return;

    if (trimmed.startsWith('/')) {
      await this._handleCommand(trimmed);
      return;
    }

    // Process user request
    try {
      console.log('');
      const response = await this.agent.processUserInput(trimmed);
      this._printResponse(response);
    } catch (err) {
      console.error('\n\x1b[31mError:\x1b[0m', err.message);
    }
  }

  async _handleCommand(cmd) {
    const parts = cmd.split(/\s+/);
    const command = parts[0].toLowerCase();

    switch (command) {
      case '/help':
        console.log(HELP_TEXT);
        break;

      case '/budget':
        if (parts[1]) {
          const n = parseInt(parts[1], 10);
          if (isNaN(n) || n < 1000) {
            console.log('\x1b[33mUsage: /budget N (N >= 1000)\x1b[0m');
          } else {
            this.agent.setBudget(n);
            console.log(`\x1b[32mBudget set to ${n.toLocaleString()} tokens\x1b[0m`);
          }
        } else {
          console.log(`\n${this.agent.budgetStatus}\n`);
        }
        break;

      case '/model':
        if (parts[1]) {
          this.agent.llm.model = parts[1];
          console.log(`\x1b[32mModel set to: ${parts[1]}\x1b[0m`);
        } else {
          console.log(`\x1b[32mCurrent model: ${this.agent.llm.model}\x1b[0m`);
        }
        break;

      case '/reset':
        this.agent.reset();
        console.log('\x1b[32mConversation reset. Budget preserved.\x1b[0m');
        break;

      case '/verbose':
        this.agent.verbose = !this.agent.verbose;
        console.log(`\x1b[32mVerbose mode: ${this.agent.verbose ? 'ON' : 'OFF'}\x1b[0m`);
        break;

      case '/exit':
        this.running = false;
        break;

      default:
        console.log(`\x1b[33mUnknown: ${command}. Type /help\x1b[0m`);
    }
  }

  _printResponse(text) {
    if (!text) return;
    console.log('');
    const lines = text.split('\n');
    for (const line of lines) {
      console.log(`  ${line}`);
    }
    console.log('');
  }
}
