import { defineTool } from "@deepseek-ai/dsh-tools";
import { SessionSeq } from "@deepseek-ai/dsh-session";

const DEFAULT_MAX_CHARS = 8000;
const MAX_SPAN = 200;

/** Flatten a derived event message into plain text for recall output. */
function messageToText(message) {
  if (typeof message === "string") return message;
  if (Array.isArray(message?.content)) {
    return message.content
      .map((block) => (typeof block === "string" ? block : block?.text ?? ""))
      .join("");
  }
  return JSON.stringify(message) ?? "";
}

export default function plugin(ctx) {
  ctx.inject(["tools"], (sctx) => {
    sctx.effect(() => {
      const dispose = sctx.tools.register(
        defineTool({
          name: "recall",
          description:
            "Returns the ORIGINAL content of session events that compaction has removed from the visible conversation, given a seq range. Seq numbers appear in compaction checkpoint messages.",
          parameters: {
            startSeq: {
              type: "number",
              required: true,
              description:
                "First event seq to recall (inclusive). Seq numbers appear in compaction checkpoint messages.",
            },
            endSeq: {
              type: "number",
              required: true,
              description:
                "Last event seq to recall (inclusive). The span from startSeq is limited to 200 seqs.",
            },
            maxChars: {
              type: "number",
              description: "Maximum total characters to return (default 8000).",
              default: DEFAULT_MAX_CHARS,
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
      return () => {
        dispose();
      };
    }, "dsh-recall: recall tool");
  });
}
