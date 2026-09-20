import { defineTool } from "@deepseek-ai/dsh-tools";
import { SessionSeq } from "@deepseek-ai/dsh-session";

const name = "recall";
const inject = ["tools"];

const DEFAULT_MAX_CHARS = 8000;
const MAX_SPAN = 200;

/** Flatten a derived event message into plain text for recall output. */
function messageToText(message) {
  const walk = (value) => {
    if (typeof value === "string") return value;
    if (Array.isArray(value)) {
      return value.map(walk).join("");
    }
    if (value !== null && typeof value === "object") {
      if (typeof value.text === "string") return value.text;
      if (Array.isArray(value.content)) return walk(value.content);
      if (Array.isArray(value.parts)) return walk(value.parts);
    }
    return "";
  };
  const text = walk(message);
  if (text === "" && message !== null && typeof message === "object") {
    return JSON.stringify(message);
  }
  return text;
}

function apply(ctx, config) {
  ctx.effect(() => {
    const dispose = ctx.tools.register(
        defineTool({
          name: "recall",
          description:
            "Returns the ORIGINAL content of session events that compaction has removed from the visible conversation, given a seq range. Call recall_index first to get the available seq ranges.",
          parameters: {
            startSeq: {
              type: "number",
              description:
                "First event seq to recall (inclusive). Seq numbers appear in compaction checkpoint messages.",
            },
            endSeq: {
              type: "number",
              description:
                "Last event seq to recall (inclusive). The span from startSeq is limited to 200 seqs.",
            },
            maxChars: {
              type: "number",
              description: "Maximum total characters to return (default 8000).",
            },
          },
          output: {
            schema: {
              type: "object",
              additionalProperties: false,
              properties: {
                recalled: { type: "number" },
                missing: { type: "number" },
                empty: { type: "number" },
                truncated: { type: "boolean" },
                content: { type: "string" },
              },
            },
            render(args, value) {
              const header = `recall: ${value.recalled} recalled, ${value.missing} missing, ${value.empty} empty, truncated=${value.truncated} (seq ${args.startSeq}..${args.endSeq})`;
              return [{ type: "text", text: `${header}\n${value.content}` }];
            },
          },
          async execute(args, exec) {
            if (!exec.agent) throw new Error("recall requires a calling agent");
            const session = exec.agent.session;
            const { startSeq, endSeq } = args;
            const maxChars = args.maxChars ?? DEFAULT_MAX_CHARS;
            if (
              startSeq < 0 ||
              endSeq < 0 ||
              !Number.isSafeInteger(startSeq) ||
              !Number.isSafeInteger(endSeq)
            ) {
              throw new Error(
                `startSeq and endSeq must be non-negative integers; got ${startSeq}, ${endSeq}`,
              );
            }
            if (startSeq > endSeq) {
              throw new Error(`startSeq (${startSeq}) must be <= endSeq (${endSeq})`);
            }
            const span = endSeq - startSeq + 1;
            if (span > MAX_SPAN) {
              throw new Error(
                `seq span must be at most ${MAX_SPAN}; got ${span} (${startSeq}..${endSeq})`,
              );
            }
            let recalled = 0;
            let missing = 0;
            let empty = 0;
            const parts = [];
            for (let seq = startSeq; seq <= endSeq; seq += 1) {
              const event = session.eventAt(SessionSeq(seq));
              if (!event) {
                missing += 1;
                continue;
              }
              const message = session.deriveEventMessage(event);
              if (message === null) {
                empty += 1;
                continue;
              }
              recalled += 1;
              parts.push({ seq, type: event.type, text: messageToText(message) });
            }
            const totalChars = parts.reduce((n, p) => n + p.text.length, 0);
            let content = "";
            let truncated = false;
            let omittedChars = 0;
            let omittedSeqs = 0;
            for (let i = 0; i < parts.length; i += 1) {
              if (content.length + parts[i].text.length > maxChars) {
                truncated = true;
                const remaining = maxChars - content.length;
                content += parts[i].text.slice(0, remaining);
                omittedChars = totalChars - content.length;
                omittedSeqs = parts.length - i;
                break;
              }
              content += parts[i].text;
            }
            if (truncated) {
              content += `\n[truncated: ${omittedChars} chars across ${omittedSeqs} seqs omitted]`;
            }
            return { recalled, missing, empty, truncated, content };
          },
        }),
      );
      const disposeIndex = ctx.tools.register(
        defineTool({
          name: "recall_index",
          description:
            "Lists the spans of this session's conversation that compaction has removed from view, so `recall` can bring them back. Call this before `recall` to find out what is available.",
          parameters: {},
          output: {
            schema: {
              type: "object",
              additionalProperties: false,
              properties: {
                spans: {
                  type: "array",
                  items: {
                    type: "object",
                    additionalProperties: false,
                    properties: {
                      kind: { type: "string" },
                      startSeq: { type: "number" },
                      endSeq: { type: "number" },
                      count: { type: "number" },
                      tokens: { type: "number" },
                    },
                  },
                },
                totalSpans: { type: "number" },
                totalTokens: { type: "number" },
              },
            },
            render(args, value) {
              const lines = [
                `recall_index: ${value.totalSpans} spans, ${value.totalTokens} tokens`,
              ];
              for (const span of value.spans) {
                lines.push(
                  `${span.kind} seq ${span.startSeq}..${span.endSeq} (${span.count} events, ${span.tokens} tokens)`,
                );
              }
              return [{ type: "text", text: lines.join("\n") }];
            },
          },
          async execute(args, exec) {
            if (!exec.agent) throw new Error("recall_index requires a calling agent");
            const session = exec.agent.session;
            const events = session.snapshotEvents();
            if (!Array.isArray(events)) {
              throw new Error(
                `recall_index: session.snapshotEvents() did not return an array (got ${typeof events})`,
              );
            }
            const spans = [];
            let totalTokens = 0;
            for (const event of events) {
              const kind =
                event?.type === "compaction/summary"
                  ? "summary"
                  : event?.type === "compaction/prune"
                    ? "prune"
                    : null;
              if (kind === null) continue;
              const shadowedSeqs = event.data?.shadowedSeqs;
              if (!Array.isArray(shadowedSeqs) || shadowedSeqs.length === 0) continue;
              const tokens = event.data?.shadowedTokenCount ?? 0;
              const span = {
                kind,
                startSeq: Math.min(...shadowedSeqs),
                endSeq: Math.max(...shadowedSeqs),
                count: shadowedSeqs.length,
                tokens,
              };
              spans.push(span);
              totalTokens += tokens;
            }
            return { spans, totalSpans: spans.length, totalTokens };
          },
        }),
      );
      return () => {
        dispose();
        disposeIndex();
      };
    }, "dsh-recall: recall and recall_index tools");
}

export { name, inject, apply };
