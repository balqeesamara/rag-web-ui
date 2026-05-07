import React, {
  FC,
  useMemo,
  useEffect,
  useState,
  useRef,
  useCallback,
  ClassAttributes,
} from "react";
import { AnchorHTMLAttributes } from "react";
import { ChevronDown, ChevronRight, Brain, Search, BookOpen, Share2 } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import { Divider } from "@/components/ui/divider";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { api } from "@/lib/api";
import { cleanChunkText } from "@/lib/utils";
import { FileIcon } from "react-file-icon";

// Debounce hook to prevent rapid state updates during streaming
const useDebouncedValue = <T,>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

const ThinkBlock: FC<{ content: string; isComplete: boolean }> = ({
  content,
  isComplete,
}) => {
  const [isExpanded, setIsExpanded] = useState(!isComplete);
  const [elapsedMs, setElapsedMs] = useState(0);
  const startTimeRef = useRef<number>(Date.now());
  const finalMsRef = useRef<number | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // Single effect: run interval while thinking, freeze + collapse when done.
  // We never call setElapsedMs synchronously inside the completion branch to
  // avoid triggering the "Maximum update depth exceeded" cascade.
  useEffect(() => {
    if (isComplete) {
      // Record final elapsed time into a ref (no setState = no re-render loop)
      if (finalMsRef.current === null) {
        finalMsRef.current = Date.now() - startTimeRef.current;
      }
      const timer = setTimeout(() => setIsExpanded(false), 1500);
      return () => clearTimeout(timer);
    }
    // Tick every 100 ms while the model is still thinking
    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTimeRef.current);
    }, 100);
    return () => clearInterval(interval);
  }, [isComplete]);

  // Auto-scroll to bottom as content streams in
  useEffect(() => {
    if (!isComplete && isExpanded && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [content, isComplete, isExpanded]);

  const displayMs = finalMsRef.current ?? elapsedMs;
  const seconds = displayMs / 1000;
  const label = isComplete
    ? seconds < 1
      ? "Thought for less than a second"
      : `Thought for ${seconds.toFixed(1)} seconds`
    : `Thinking... (${seconds.toFixed(1)}s)`;

  return (
    <div className="my-2 rounded-md border border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 w-full">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-left rounded-t-md hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors group"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-gray-400 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-gray-400 shrink-0" />
        )}
        <Brain className={`h-3 w-3 shrink-0 ${isComplete ? "text-gray-400" : "text-blue-400 animate-pulse"}`} />
        <span className="text-xs text-gray-400 font-medium select-none">
          {label}
        </span>
      </button>
      {isExpanded && (
        <div
          ref={contentRef}
          className="px-3 pb-2 pt-1 max-h-48 overflow-y-auto overflow-x-hidden border-t border-gray-100 dark:border-gray-700"
        >
          <pre className="text-[11px] leading-[1.45] text-gray-400 dark:text-gray-500 whitespace-pre-wrap break-words font-sans m-0">
            {content}
          </pre>
        </div>
      )}
    </div>
  );
};

interface ContextDoc {
  page_content: string;
  metadata: Record<string, any>;
}

const RewrittenQueryBlock: FC<{ query: string }> = ({ query }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setIsExpanded(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="my-2 rounded-md border border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 w-full">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-left rounded-t-md hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-gray-400 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-gray-400 shrink-0" />
        )}
        <Search className="h-3 w-3 shrink-0 text-gray-400" />
        <span className="text-xs text-gray-400 font-medium select-none">
          Rewritten Query
        </span>
      </button>
      {isExpanded && (
        <div className="px-3 pb-2 pt-1 border-t border-gray-100 dark:border-gray-700">
          <p className="text-[11px] leading-[1.45] text-gray-400 dark:text-gray-500 whitespace-pre-wrap break-words font-sans m-0">
            {query}
          </p>
        </div>
      )}
    </div>
  );
};

const RetrievedContextBlock: FC<{ docs: ContextDoc[] }> = ({ docs }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="my-2 rounded-md border border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 w-full">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-left rounded-t-md hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-gray-400 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-gray-400 shrink-0" />
        )}
        <BookOpen className="h-3 w-3 shrink-0 text-gray-400" />
        <span className="text-xs text-gray-400 font-medium select-none">
          Retrieved {docs.length} context{docs.length !== 1 ? "s" : ""}
        </span>
      </button>
      {isExpanded && (
        <div className="px-3 pb-2 pt-1 max-h-64 overflow-y-auto border-t border-gray-100 dark:border-gray-700 space-y-2">
          {docs.map((doc, i) => (
            <div key={i} className="text-[11px] leading-[1.45] text-gray-400 dark:text-gray-500 font-sans">
              <span className="font-semibold text-gray-500 dark:text-gray-400">[{i + 1}] </span>
              {doc.page_content.length > 300
                ? doc.page_content.slice(0, 300) + "..."
                : doc.page_content}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

interface Citation {
  id: number;
  text: string;
  metadata: Record<string, any>;
}

interface KnowledgeBaseInfo {
  name: string;
}

interface DocumentInfo {
  file_name: string;
  knowledge_base: KnowledgeBaseInfo;
}

const RetrievedGraphBlock: FC<{ docs: ContextDoc[] }> = ({ docs }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="my-2 rounded-md border border-purple-100 dark:border-purple-900/40 bg-purple-50/50 dark:bg-purple-900/10 w-full">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-left rounded-t-md hover:bg-purple-100/60 dark:hover:bg-purple-900/20 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-purple-400 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-purple-400 shrink-0" />
        )}
        <Share2 className="h-3 w-3 shrink-0 text-purple-400" />
        <span className="text-xs text-purple-500 dark:text-purple-400 font-medium select-none">
          Retrieved Graph Knowledge
        </span>
        <span className="ml-auto text-[10px] text-purple-400 dark:text-purple-500 font-normal select-none">
          {docs.length} node{docs.length !== 1 ? "s" : ""}
        </span>
      </button>
      {isExpanded && (
        <div className="px-3 pb-2 pt-1 max-h-64 overflow-y-auto border-t border-purple-100 dark:border-purple-900/40 space-y-2">
          {docs.map((doc, i) => (
            <div key={i} className="text-[11px] leading-[1.45] text-purple-700 dark:text-purple-300 font-sans">
              <span className="font-semibold text-purple-500 dark:text-purple-400">[G{i + 1}] </span>
              <span className="whitespace-pre-wrap">
                {cleanChunkText(doc.page_content).length > 400
                  ? cleanChunkText(doc.page_content).slice(0, 400) + "…"
                  : cleanChunkText(doc.page_content)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

interface CitationInfo {
  knowledge_base: KnowledgeBaseInfo;
  document: DocumentInfo;
}

// ── Confidence bar ─────────────────────────────────────────────────────────────

type ConfidenceLevel = "very_high" | "high" | "medium" | "low" | "none";

const CONFIDENCE_CONFIG: Record<ConfidenceLevel, {
  steps: number;   // how many of 4 steps are filled
  label: string;
  stepColor: string;
  textColor: string;
  bgColor: string;
  borderColor: string;
}> = {
  very_high: { steps: 4, label: "Very High",  stepColor: "bg-emerald-500", textColor: "text-emerald-700", bgColor: "bg-emerald-50",  borderColor: "border-emerald-200" },
  high:      { steps: 3, label: "High",       stepColor: "bg-green-500",   textColor: "text-green-700",   bgColor: "bg-green-50",    borderColor: "border-green-200"   },
  medium:    { steps: 2, label: "Medium",     stepColor: "bg-yellow-500",  textColor: "text-yellow-700",  bgColor: "bg-yellow-50",   borderColor: "border-yellow-200"  },
  low:       { steps: 1, label: "Low",        stepColor: "bg-orange-500",  textColor: "text-orange-700",  bgColor: "bg-orange-50",   borderColor: "border-orange-200"  },
  none:      { steps: 0, label: "None",       stepColor: "bg-red-400",     textColor: "text-red-700",     bgColor: "bg-red-50",      borderColor: "border-red-200"     },
};

const ConfidenceBar: FC<{
  level: ConfidenceLevel;
  score?: number;
  suggestion?: string | null;
}> = ({ level, score, suggestion }) => {
  const cfg = CONFIDENCE_CONFIG[level];
  return (
    <div className={`rounded-md border ${cfg.borderColor} ${cfg.bgColor} px-3 py-2 mb-3 not-prose`}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className={`text-xs font-medium ${cfg.textColor} shrink-0`}>
            Retrieval confidence
          </span>
          <span className={`text-xs font-semibold ${cfg.textColor} shrink-0`}>
            {cfg.label}{score !== undefined ? ` · ${score}/100` : ""}
          </span>
        </div>
        {/* stepped progress bar — 4 equal segments */}
        <div className="flex gap-1 shrink-0">
          {[1, 2, 3, 4].map(step => (
            <div
              key={step}
              className={`h-2 w-7 rounded-sm transition-colors ${
                step <= cfg.steps ? cfg.stepColor : "bg-zinc-200"
              }`}
            />
          ))}
        </div>
      </div>
      {suggestion && (
        <p className={`mt-1 text-xs ${cfg.textColor} opacity-80`}>{suggestion}</p>
      )}
    </div>
  );
};

export const Answer: FC<{
  markdown: string;
  citations?: Citation[];
  rewrittenQuery?: string;
  retrievedContext?: ContextDoc[];
  confidence?: "very_high" | "high" | "medium" | "low" | "none";
  confidenceScore?: number;
  suggestion?: string | null;
}> = ({ markdown, citations = [], rewrittenQuery, retrievedContext, confidence, confidenceScore, suggestion }) => {
  const [citationInfoMap, setCitationInfoMap] = useState<
    Record<string, CitationInfo>
  >({});

  // Debounce citations to prevent rapid API calls during streaming
  const debouncedCitations = useDebouncedValue(citations, 300);

  // Keep refs so CitationLink can read the latest data without changing its
  // identity (avoiding react-markdown remounting all <a> elements every render)
  const citationsRef = useRef(debouncedCitations);
  const citationInfoMapRef = useRef(citationInfoMap);
  citationsRef.current = debouncedCitations;
  citationInfoMapRef.current = citationInfoMap;

  const parsedContent = useMemo(() => {
    // Non-anchored: handles models that emit text before <think> (preamble)
    const completeMatch = markdown.match(/([\s\S]*?)<think>([\s\S]*?)<\/think>([\s\S]*)$/);
    if (completeMatch) {
      const preamble = completeMatch[1];
      const thinkContent = completeMatch[2].trim();
      const afterThink = completeMatch[3].trim();
      return {
        thinkContent,
        isThinkingComplete: true,
        // Preserve any preamble text before the <think> block
        answerText: preamble ? `${preamble.trim()}\n\n${afterThink}`.trim() : afterThink,
      };
    }
    // <think> opened but not yet closed — still streaming
    const openMatch = markdown.match(/([\s\S]*?)<think>([\s\S]*)$/);
    if (openMatch) {
      const preamble = openMatch[1];
      return {
        thinkContent: openMatch[2],
        isThinkingComplete: false,
        answerText: preamble.trim(),
      };
    }
    return { thinkContent: null, isThinkingComplete: false, answerText: markdown };
  }, [markdown]);

  useEffect(() => {
    const fetchCitationInfo = async () => {
      const infoMap: Record<string, CitationInfo> = {};

      for (const citation of debouncedCitations) {
        const { kb_id, document_id } = citation.metadata;
        if (!kb_id || !document_id) continue;

        const key = `${kb_id}-${document_id}`;
        if (infoMap[key]) continue;

        try {
          const [kb, doc] = await Promise.all([
            api.get(`/api/knowledge-base/${kb_id}`),
            api.get(`/api/knowledge-base/${kb_id}/documents/${document_id}`),
          ]);

          infoMap[key] = {
            knowledge_base: {
              name: kb.name,
            },
            document: {
              file_name: doc.file_name,
              knowledge_base: {
                name: kb.name,
              },
            },
          };
        } catch (error) {
          console.error("Failed to fetch citation info:", error);
        }
      }

      setCitationInfoMap(infoMap);
    };

    if (debouncedCitations.length > 0) {
      fetchCitationInfo();
    }
  }, [debouncedCitations]);

  // Stable component reference — never recreated, reads current data from refs.
  // This prevents react-markdown from unmounting/remounting all <a> elements
  // whenever citationInfoMap or debouncedCitations change, which was causing
  // Radix Popover state cascades and "Maximum update depth exceeded".
  const CitationLink = useCallback(
    (
      props: ClassAttributes<HTMLAnchorElement> &
        AnchorHTMLAttributes<HTMLAnchorElement>
    ) => {
      const citationId = props.href?.match(/^(\d+)$/)?.[1];
      const citation = citationId
        ? citationsRef.current[parseInt(citationId) - 1]
        : null;

      if (!citation) {
        return <a>[{props.href}]</a>;
      }

      const citationInfo =
        citationInfoMapRef.current[
          `${citation.metadata.kb_id}-${citation.metadata.document_id}`
        ];

      return (
        <Popover>
          <PopoverTrigger asChild>
            <a
              {...props}
              href="#"
              role="button"
              className="inline-flex items-center gap-1 px-1.5 py-0.5 text-xs font-medium text-blue-600 bg-blue-50 rounded hover:bg-blue-100 transition-colors relative"
            >
              <span className="absolute -top-3 -right-1">[{props.href}]</span>
            </a>
          </PopoverTrigger>
          <PopoverContent
            side="top"
            align="start"
            className="max-w-2xl w-[calc(100vw-100px)] p-4 rounded-lg shadow-lg"
          >
            <div className="text-sm space-y-3">
              {citationInfo && (
                <div className="flex items-center gap-2 text-xs font-medium text-gray-700 bg-gray-50 p-2 rounded">
                  <div className="w-5 h-5 flex items-center justify-center">
                    <FileIcon
                      extension={
                        citationInfo.document.file_name.split(".").pop() || ""
                      }
                      color="#E2E8F0"
                      labelColor="#94A3B8"
                    />
                  </div>
                  <span className="truncate">
                    {citationInfo.knowledge_base.name} /{" "}
                    {citationInfo.document.file_name}
                  </span>
                </div>
              )}
              <Divider />
              <div className="text-gray-700 leading-relaxed prose prose-sm dark:prose-invert max-w-none">
                <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                  {cleanChunkText(citation.text)}
                </Markdown>
              </div>
              <Divider />
              {Object.keys(citation.metadata).length > 0 && (
                <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                  <div className="font-medium mb-2">Debug Info:</div>
                  <div className="space-y-1">
                    {Object.entries(citation.metadata).map(([key, value]) => (
                      <div key={key} className="flex">
                        <span className="font-medium min-w-[100px]">
                          {key}:
                        </span>
                        <span className="text-gray-600">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </PopoverContent>
        </Popover>
      );
    },
    [] // stable — reads from refs
  );

  // Memoize the components object so react-markdown never sees a new reference
  const markdownComponents = useMemo(() => ({ a: CitationLink }), [CitationLink]);

  // Key changes only when citation info is first fetched; this forces a single
  // controlled remount of <Markdown> (so popover content updates after the
  // async fetch), instead of continuous uncontrolled remounts during streaming.
  const citationInfoKey = Object.keys(citationInfoMap).sort().join(",");

  if (!markdown && !rewrittenQuery && (!retrievedContext || retrievedContext.length === 0)) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="max-w-sm h-4 bg-zinc-200" />
        <Skeleton className="max-w-lg h-4 bg-zinc-200" />
        <Skeleton className="max-w-2xl h-4 bg-zinc-200" />
        <Skeleton className="max-w-lg h-4 bg-zinc-200" />
        <Skeleton className="max-w-xl h-4 bg-zinc-200" />
      </div>
    );
  }

  return (
    <div className="prose prose-sm max-w-full">
      {rewrittenQuery && <RewrittenQueryBlock query={rewrittenQuery} />}
      {confidence && confidence !== "none" && (
        <ConfidenceBar level={confidence} score={confidenceScore} suggestion={suggestion} />
      )}
      {confidence === "none" && suggestion && (
        <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 mb-2">
          <span className="mt-0.5 shrink-0">⚠</span>
          <span>{suggestion}</span>
        </div>
      )}
      {retrievedContext && retrievedContext.filter(d => d.metadata?.source !== "graph").length > 0 && (
        <RetrievedContextBlock docs={retrievedContext.filter(d => d.metadata?.source !== "graph")} />
      )}
      {retrievedContext && retrievedContext.filter(d => d.metadata?.source === "graph").length > 0 && (
        <RetrievedGraphBlock docs={retrievedContext.filter(d => d.metadata?.source === "graph")} />
      )}
      {!markdown && (
        <div className="flex flex-col gap-2 mt-2">
          <Skeleton className="max-w-sm h-4 bg-zinc-200" />
          <Skeleton className="max-w-lg h-4 bg-zinc-200" />
          <Skeleton className="max-w-2xl h-4 bg-zinc-200" />
        </div>
      )}
      {parsedContent.thinkContent !== null && (
        <ThinkBlock
          content={parsedContent.thinkContent}
          isComplete={parsedContent.isThinkingComplete}
        />
      )}
      {parsedContent.answerText && (
        <Markdown
          key={citationInfoKey}
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={markdownComponents}
        >
          {parsedContent.answerText}
        </Markdown>
      )}
    </div>
  );
};
