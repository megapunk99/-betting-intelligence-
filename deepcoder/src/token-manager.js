/**
 * Token Manager - enforces budgets so the agent doesn't burn through tokens.
 *
 * Architecture:
 * - sessionBudget: max tokens allowed for the entire session (default 100K)
 * - turnBudget: max tokens allowed for a single agentic turn (default 20K)
 * - runningTotal: tokens consumed so far
 * - contextMessages: the active message array sent to the LLM
 */
export class TokenManager {
  constructor(options = {}) {
    this.sessionBudget = options.sessionBudget ?? 100_000;   // 100K tokens per session
    this.turnBudget = options.turnBudget ?? 20_000;          // 20K tokens per turn
    this.runningTotal = 0;
    this.contextMessages = [];
    this._onBudgetExhausted = options.onBudgetExhausted || null;
  }

  /** Register tokens spent and check budgets */
  spend(count, { source = 'unknown' } = {}) {
    this.runningTotal += count;
    if (this.runningTotal >= this.sessionBudget) {
      if (this._onBudgetExhausted) this._onBudgetExhausted();
      throw new TokenBudgetError(
        `Session budget exhausted (${this.runningTotal}/${this.sessionBudget} tokens). ` +
        'Use /budget to increase the limit or start a new session.'
      );
    }
  }

  /** How many tokens remain in the session budget */
  get remaining() {
    return Math.max(0, this.sessionBudget - this.runningTotal);
  }

  /** How many tokens remain for the current turn */
  get turnRemaining() {
    return Math.max(0, this.turnBudget - this._turnUsed);
  }

  /** Rough estimate of how many tokens a string consumes */
  static estimate(text) {
    if (!text) return 0;
    // ~4 chars per token for English text, slightly worse for code
    return Math.ceil(text.length / 2.8);
  }

  /** Build a compact context by pruning older messages when nearing budget */
  buildContext(messages) {
    let total = 0;
    const pruned = [];

    // Always keep the system message
    const systemMsg = messages.find(m => m.role === 'system');
    if (systemMsg) {
      pruned.push(systemMsg);
      total += TokenManager.estimate(systemMsg.content);
    }

    // Keep the most recent messages that fit within turn budget
    const nonSystem = messages.filter(m => m.role !== 'system');

    // Always keep the last user message (the current request) regardless of size
    const lastMsg = nonSystem.length > 0 ? nonSystem[nonSystem.length - 1] : null;
    const lastMsgCost = lastMsg ? TokenManager.estimate(lastMsg.content) : 0;
    total += lastMsgCost;

    // Work backwards from second-to-last, keeping what fits
    const toKeep = [];
    if (lastMsg) toKeep.push(lastMsg);

    for (let i = nonSystem.length - 2; i >= 0; i--) {
      const cost = TokenManager.estimate(nonSystem[i].content);
      if (total + cost > this.turnBudget * 0.85) {
        break;
      }
      toKeep.push(nonSystem[i]);
      total += cost;
    }

    // Reorder back to chronological
    const result = [pruned[0], ...toKeep.reverse()];

    this.contextMessages = result;
    return result;
  }

  /** Format a human-readable budget status */
  status() {
    const pct = ((this.runningTotal / this.sessionBudget) * 100).toFixed(1);
    return [
      `Budget: ${this.runningTotal.toLocaleString()} / ${this.sessionBudget.toLocaleString()} tokens used (${pct}%)`,
      `Remaining: ${this.remaining.toLocaleString()} tokens`,
      `Turn limit: ${this.turnBudget.toLocaleString()} tokens`
    ].join('\n');
  }

  /** Reset counters (for a new session) */
  reset() {
    this.runningTotal = 0;
    this.contextMessages = [];
  }
}

export class TokenBudgetError extends Error {
  constructor(message) {
    super(message);
    this.name = 'TokenBudgetError';
  }
}
