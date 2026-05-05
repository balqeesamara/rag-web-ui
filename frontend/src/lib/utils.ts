import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Clean raw chunk text for display in citation popups.
 *
 * PDF/OCR extraction produces heavily wrapped text: lines break every ~40 chars
 * due to column layout, and blank lines appear between sentence fragments.
 * This function reflows everything into proper prose/list paragraphs.
 *
 * Strategy:
 *  1. Strip markitdown OCR image blocks entirely (noise)
 *  2. Convert ● bullets to markdown list items
 *  3. Flatten all lines into a stream; classify each as:
 *     - section heading: short, starts uppercase, no trailing punctuation,
 *                        doesn't start with a conjunction/preposition
 *     - list item: starts with -
 *     - prose fragment: everything else → join to previous line with space
 */

// Words that begin sentence fragments, not section titles
const FRAG_STARTER =
  /^(and|or|but|with|for|to|in|of|the|a|an|by|from|at|as|on|be|is|are|was|were|that|this|it|if|when|while|since|after|before|because|although|however|therefore|thus|so|yet|nor|than|unless|until|though|we|our|i|you|he|she|they)\b/i;

function looksLikeHeading(line: string): boolean {
  if (line.length > 60) return false;               // too long for a heading
  if (/[.!?)\]>]$/.test(line)) return false;        // ends with sentence punctuation
  if (/^[-*#|`~]/.test(line)) return false;          // markdown / list / code
  if (!/^[A-Z0-9(~]/.test(line)) return false;      // must start uppercase, digit, or symbol
  if (FRAG_STARTER.test(line)) return false;         // fragment continuation
  return true;
}

export function cleanChunkText(text: string): string {
  // 1. Strip *[Image OCR] ... [End OCR]* blocks
  let s = text.replace(/\*\[Image OCR\][\s\S]*?\[End OCR\]\*/g, "").trim();

  // 2. ● → markdown list item
  s = s.replace(/●\s*/g, "- ");

  // 3. Flatten and classify line by line
  const lines = s
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l !== "");

  const output: string[] = [];
  let buffer = "";
  let inList = false;

  const flush = () => {
    if (buffer) {
      output.push(buffer);
      buffer = "";
    }
  };

  for (const line of lines) {
    const isListItem = /^[-*]\s/.test(line);
    const isMarkdownHeading = /^#{1,6}\s/.test(line);

    if (isMarkdownHeading) {
      flush();
      inList = false;
      output.push(line);
    } else if (looksLikeHeading(line)) {
      const lastOut = output[output.length - 1];
      // Dangling fragment: short, ends with punctuation, follows prose
      const looksLikeDanglingFragment =
        line.length < 30 &&
        /[.!?]$/.test(line) &&
        lastOut &&
        !lastOut.startsWith("**") &&
        !lastOut.startsWith("-");
      // Heading continuation: previous output was a bold heading — merge
      // the two lines into one (removes the newline within a heading)
      const isPrevHeading = lastOut?.startsWith("**") && lastOut.endsWith("**");
      // Only merge if previous heading looks genuinely incomplete:
      // ends with a hyphen (word-split) or is very short (≤25 chars of content)
      const prevContent = isPrevHeading
        ? lastOut.replace(/^\*\*/, "").replace(/\*\*$/, "")
        : "";
      const prevHeadingIncomplete =
        isPrevHeading && prevContent.endsWith("-");
      if (looksLikeDanglingFragment) {
        output[output.length - 1] += " " + line;
      } else if (prevHeadingIncomplete && !buffer) {
        // Merge: strip trailing ** from previous, append, re-wrap
        output[output.length - 1] =
          output[output.length - 1].replace(/\*\*$/, "") +
          " " + line + "**";
      } else {
        flush();
        inList = false;
        output.push(`**${line}**`);
      }
    } else if (isListItem) {
      if (!inList && buffer) flush();
      if (inList && buffer) flush();
      buffer = line;
      inList = true;
    } else {
      if (buffer) buffer += " " + line;
      else buffer = line;
    }
  }
  flush();

  return output.join("\n\n").trim();
}
