/**
 * Agent — text-based command protocol.
 *
 * How it works:
 * 1. User sends a request
 * 2. We call the LLM with system prompt + conversation
 * 3. LLM responds with text that may contain commands like [READ path], [EDIT path], etc.
 * 4. We parse commands, execute them, and show results
 * 5. We call the LLM again with the results
 * 6. Loop until LLM says [DONE] or max turns reached
 *
 * This approach is:
 * - Model-agnostic (works even if model doesn't support function calling)
 * - More token-efficient (no tool call JSON overhead)
 * - More robust (simple regex parsing)
 */
import { readFile } from 'node:fs/promises';
import { writeFile, mkdir } from 'node:fs/promises';
import { readdir } from 'node:fs/promises';
import { statSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { TokenManager } from './token-manager.js';
import { LLMClient } from './llm.js';
import { SYSTEM_PROMPT } from './prompts.js';

// ─── Command regex patterns ───────────────────────────────────────
const CMD_PATTERNS = {
  // Model outputs: [READ path/to/file] — arg is inside brackets
  READ:  /^\[READ\s+(.+?)\]$/im,
  // [WRITE path]\ncontent\n---
  WRITE: /^\[WRITE\s+(.+?)\]$\n([\s\S]*?)\n?^---$/im,
  // [EDIT path]\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>>
  EDIT:  /^\[EDIT\s+(.+?)\]$\n<<<<<<< SEARCH\n([\s\S]*?)=======\n([\s\S]*?)>>>>>>>\n?/gm,
  SEARCH:/^\[SEARCH\s+(.+?)\]$/im,
  GLOB:  /^\[GLOB\s+(.+?)\]$/im,
  LIST:  /^\[LIST\s+(.+?)\]$/im,
  BASH:  /^\[BASH\s+(.+?)\]$/im,
  DONE:  /^\s*\[DONE\]\s*$/im,
};

export class Agent {
  constructor(options = {}) {
    this.llm = new LLMClient(options);
    this.tokenManager = new TokenManager(options);
    this.maxTurns = options.maxTurns ?? 15;          // Max LLM calls per user request
    this.maxToolCalls = options.maxToolCalls ?? 30;   // Max total commands per user request
    this.messages = [];
    this.cwd = options.cwd || process.cwd();
    this.verbose = options.verbose ?? false;

    this.messages.push({ role: 'system', content: SYSTEM_PROMPT });
  }

  /** Main entry: process a user message */
  async processUserInput(userInput) {
    this.messages.push({ role: 'user', content: userInput });
    const response = await this._agenticLoop();
    this.messages.push({ role: 'assistant', content: response });
    return response;
  }

  /** Agentic loop: LLM → parse commands → execute → LLM → ... */
  async _agenticLoop() {
    let turnCount = 0;
    let toolCount = 0;
    let lastResponse = '';
    let lastResultSummary = '';

    while (turnCount < this.maxTurns && toolCount < this.maxToolCalls) {
      turnCount++;

      // Build context (prune old messages to fit budget)
      const contextMessages = this.tokenManager.buildContext(this.messages);

      // Call LLM
      let response;
      try {
        response = await this.llm.complete(contextMessages, {
          maxTokens: Math.min(this.tokenManager.turnBudget, 4096),
        });
      } catch (err) {
        return `⚠️  API error: ${err.message}\n\nTry again or check your API key and network connection.`;
      }

      // Track tokens
      if (response.usage) {
        const total = (response.usage.prompt_tokens || 0) + (response.usage.completion_tokens || 0);
        this.tokenManager.spend(total);
      }

      const text = response.choices?.[0]?.message?.content || '';
      if (!text) return '⚠️  Empty response from model.';

      if (this.verbose) {
        console.error(`\n\x1b[90m[Turn ${turnCount}] LLM response (${text.length} chars):\x1b[0m`);
      }

      // Parse commands FIRST (before checking [DONE])
      const commands = this._parseCommands(text);

      // If there are commands to execute, do them even if [DONE] is also present
      // Only check [DONE] when there are NO commands left to execute
      if (commands.length === 0 && CMD_PATTERNS.DONE.test(text)) {
        // Extract summary (text before [DONE] marker)
        const parts = text.split(/\[DONE\]/i);
        const beforeDone = parts[0].trim();
        // If there's text before [DONE], use it; otherwise check after
        lastResponse = beforeDone || parts.slice(1).join(' ').trim() || lastResultSummary || 'Task complete.';
        if (this.verbose) console.error('\x1b[32m[DONE] signal received.\x1b[0m');
        break;
      }

      if (commands.length === 0) {
        // LLM is just thinking/responding — no commands yet, feed it back
        lastResponse = text;
        if (turnCount >= this.maxTurns) break;
        // Add the response so the LLM sees its own reasoning
        this.messages.push({ role: 'assistant', content: text });
        this.messages.push({
          role: 'user',
          content: 'Continue. Issue commands or write [DONE] when finished.',
        });
        continue;
      }

      // Execute commands and collect results
      const results = [];
      for (const cmd of commands) {
        if (toolCount >= this.maxToolCalls) {
          results.push(`⚠️  Max commands reached (${this.maxToolCalls}). Stopping.`);
          break;
        }
        toolCount++;

        try {
          const result = await this._executeCommand(cmd);
          results.push(result);
        } catch (err) {
          results.push(`⚠️  Command error [${cmd.type}]: ${err.message}`);
        }
      }

      // Show thinking text before results (if any)
      const thinkingText = this._extractThinking(text, commands);
      const resultBlock = results.join('\n\n');
      lastResultSummary = resultBlock.slice(0, 500);

      // Feed results back to LLM
      // IMPORTANT: Only store the thinking text, NOT the commands themselves
      // This saves tokens and prevents the LLM from re-executing old commands
      const storedText = thinkingText || '(executing commands...) ' + commands.map(c => c.type).join(', ');
      this.messages.push({ role: 'assistant', content: storedText });
      this.messages.push({
        role: 'user',
        content: `Results:\n\n${resultBlock}\n\nNow provide your answer based on these results. Then write [DONE] followed by a brief summary.`,
      });
    }

    return lastResponse || 'Reached maximum turns. Please provide next instructions.';
  }

  /** Parse commands from LLM text output */
  _parseCommands(text) {
    const commands = [];

    // EDIT commands (multi-line, need special handling)
    let match;
    const editRegex = new RegExp(CMD_PATTERNS.EDIT.source, 'gm');
    while ((match = editRegex.exec(text)) !== null) {
      commands.push({
        type: 'EDIT',
        path: match[1].trim(),
        oldString: match[2],
        newString: match[3],
      });
    }

    // WRITE commands (multi-line)
    const writeRegex = new RegExp(CMD_PATTERNS.WRITE.source, 'gm');
    while ((match = writeRegex.exec(text)) !== null) {
      commands.push({
        type: 'WRITE',
        path: match[1].trim(),
        content: match[2].trim(),
      });
    }

    // Single-line commands
    const lineCommands = ['READ', 'SEARCH', 'GLOB', 'LIST', 'BASH'];
    for (const type of lineCommands) {
      const regex = new RegExp(CMD_PATTERNS[type].source, 'gm');
      while ((match = regex.exec(text)) !== null) {
        commands.push({ type, arg: match[1].trim() });
      }
    }

    return commands;
  }

  /** Extract thinking text (text before first command) */
  _extractThinking(text, commands) {
    if (commands.length === 0) return text;
    // Match [COMMAND (with any content inside brackets) or [COMMAND] with no args
    const firstCmdIndex = text.search(/\[(?:READ|WRITE|EDIT|SEARCH|GLOB|LIST|BASH|DONE)/);
    if (firstCmdIndex > 0) {
      return text.slice(0, firstCmdIndex).trim();
    }
    return '';
  }

  /** Execute a single parsed command */
  async _executeCommand(cmd) {
    switch (cmd.type) {
      case 'READ':
        return await this._cmdRead(cmd.arg);
      case 'WRITE':
        return await this._cmdWrite(cmd.path, cmd.content);
      case 'EDIT':
        return await this._cmdEdit(cmd.path, cmd.oldString, cmd.newString);
      case 'SEARCH':
        return await this._cmdSearch(cmd.arg);
      case 'GLOB':
        return await this._cmdGlob(cmd.arg);
      case 'LIST':
        return await this._cmdList(cmd.arg);
      case 'BASH':
        return await this._cmdBash(cmd.arg);
      default:
        return `Unknown command: ${cmd.type}`;
    }
  }

  // ─── Command implementations ──────────────────────────────────

  async _cmdRead(path) {
    try {
      const resolved = resolve(this.cwd, path);
      const content = await readFile(resolved, 'utf-8');
      return `[${path}]\n\`\`\`\n${content}\n\`\`\``;
    } catch (err) {
      return `⚠️  Cannot read ${path}: ${err.message}`;
    }
  }

  async _cmdWrite(path, content) {
    try {
      const resolved = resolve(this.cwd, path);
      await mkdir(dirname(resolved), { recursive: true });
      await writeFile(resolved, content, 'utf-8');
      return `✅ Wrote ${path} (${content.length} chars)`;
    } catch (err) {
      return `⚠️  Cannot write ${path}: ${err.message}`;
    }
  }

  async _cmdEdit(path, oldString, newString) {
    try {
      const resolved = resolve(this.cwd, path);
      let content = await readFile(resolved, 'utf-8');

      // Count and replace all occurrences
      let count = 0;
      let idx = content.indexOf(oldString);
      if (idx === -1) {
        return `⚠️  Cannot edit ${path}: search string not found. The file may have changed.`;
      }
      while (idx !== -1) {
        content = content.slice(0, idx) + newString + content.slice(idx + oldString.length);
        count++;
        // Search again from after the replacement
        idx = content.indexOf(oldString, idx + newString.length);
      }

      await writeFile(resolved, content, 'utf-8');
      return count > 1
        ? `✅ Edited ${path} (${count} occurrences)`
        : `✅ Edited ${path}`;
    } catch (err) {
      return `⚠️  Cannot edit ${path}: ${err.message}`;
    }
  }

  async _cmdSearch(pattern) {
    try {
      // Windows-friendly search: try ripgrep → grep → findstr on all files
      const cmd = `rg --no-heading -n -m 15 -- "${pattern}" . 2>/dev/null || grep -rns "${pattern}" . 2>/dev/null || (findstr /snip "${pattern}" . 2>nul)`;
      const output = execSync(cmd, { cwd: this.cwd, encoding: 'utf-8', timeout: 15000 });
      const lines = output.trim().split('\n').filter(Boolean).slice(0, 30);
      if (lines.length === 0) return 'No matches found.';
      return `Matches for "${pattern}":\n${lines.join('\n')}`;
    } catch {
      return 'No matches found.';
    }
  }

  async _cmdGlob(pattern) {
    try {
      const cmd = `find . -path "./node_modules" -prune -o -name "${pattern}" -print 2>/dev/null || (dir /s /b ${pattern} 2>nul)`;
      const output = execSync(cmd, { cwd: this.cwd, encoding: 'utf-8', timeout: 10000 });
      const files = output.trim().split('\n').filter(Boolean).slice(0, 30);
      if (files.length === 0) return 'No files match pattern.';
      return `Files matching "${pattern}":\n${files.join('\n')}`;
    } catch {
      return 'No files match pattern.';
    }
  }

  async _cmdList(path) {
    try {
      const resolved = resolve(this.cwd, path);
      const entries = await readdir(resolved, { withFileTypes: true });
      const files = entries.filter(e => e.isFile()).map(e => e.name);
      const dirs = entries.filter(e => e.isDirectory()).map(e => e.name + '/');
      let result = `[${path}]\n`;
      if (dirs.length) result += `  ${dirs.join('  ')}\n`;
      if (files.length) result += `  ${files.join('  ')}`;
      return result;
    } catch (err) {
      return `⚠️  Cannot list ${path}: ${err.message}`;
    }
  }

  async _cmdBash(command) {
    // Safety check
    const dangerous = /^rm\s+-rf/i.test(command.trim());
    if (dangerous) {
      return `⚠️  Dangerous command blocked: ${command}. Run it manually if needed.`;
    }

    try {
      const output = execSync(command, {
        cwd: this.cwd,
        encoding: 'utf-8',
        timeout: 30000,
        maxBuffer: 5 * 1024 * 1024,
      });
      const stdout = output.trim();
      if (!stdout) return 'Command completed (no output).';
      // Truncate to 2000 chars to save tokens
      return stdout.length > 2000
        ? stdout.slice(0, 2000) + '\n... (truncated)'
        : stdout;
    } catch (err) {
      const stderr = err.stderr?.toString().trim() || '';
      const stdout = err.stdout?.toString().trim() || '';
      return `Exit code ${err.status}: ${stderr || stdout || err.message}`;
    }
  }

  // ─── Helpers ───────────────────────────────────────────────────

  get budgetStatus() {
    return this.tokenManager.status();
  }

  setBudget(tokens) {
    this.tokenManager.sessionBudget = tokens;
  }

  reset() {
    this.messages = [{ role: 'system', content: SYSTEM_PROMPT }];
    this.tokenManager.reset();
  }
}
