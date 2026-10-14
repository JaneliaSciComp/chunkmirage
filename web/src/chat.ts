// The code panel's chat: ask a model for the Python, with the visitor's own key. Deep Chat
// draws the conversation and calls the provider from this tab, through its own connection to
// each (their APIs differ); the page adds what it knows of the layer to each message and takes
// the code from each reply, into the editor, where it runs as code typed there does. Keys are
// kept in this module only (not in the page's storage, which a link's Python could read), for
// this visit, and go nowhere but to the provider picked.
import "deep-chat";  // defines <deep-chat>
import type { DeepChat } from "deep-chat";
import type { DirectConnection } from "deep-chat/dist/types/directConnection";
import type { RequestDetails } from "deep-chat/dist/types/interceptors";

/** What the model is told once: the python op as the page runs it. */
const SYSTEM = `You write Python for chunkmirage's \`python\` op, for a visitor of a web page showing an \
image (3-D, or 2-D over time) in Neuroglancer. The page runs the code with Pyodide on each chunk of \
one layer as the viewer asks for it, at every resolution level it shows, and the layer shows what \
the code returns. The visitor's latest message starts with a <context> block: the layer, its data \
and the code as it is now.

The code defines \`apply(block, ...)\`:
- \`block\` is one chunk as a numpy array, with the axes the context gives, padded by the halo on \
each side. apply returns the chunk computed, either the padded shape or the interior (the halo \
shaved off each side).
- numpy is \`np\`, already imported. scipy and scikit-image can be imported; nothing else, and no \
files or network.
- Its other parameters are the sliders' values, by name; a parameter named \`box\` gets where the \
block sits (rarely needed).
- What it prints shows under the code, by level and chunk.

Lines read before it runs, at the top:
- \`#halo 3\` or one per axis, \`#halo 0 3 3\`: voxels of context on each side (filters and \
morphology need it, about 3 sigma for a Gaussian).
- \`#output uint32 label\`: the dtype returned (uint8, uint16, uint32, float32, ...) and/or image, \
label or mask. Connected components and other labels return \`#output uint32 label\`, and the layer \
becomes a segmentation. Without it the output has the input's dtype, so a float result needs \
\`#output float32\`.
- \`#uicontrol float low slider(min=0, max=255, step=1, default=128)\`, \`#uicontrol int size \
slider(min=0, max=500, default=50)\` or \`#uicontrol bool fill checkbox(default=true)\`: sliders \
and checkboxes under the code; apply takes a parameter of the same name. Give thresholds, sizes \
and sigmas the visitor may tune sliders, with defaults that suit the data's range.

Each chunk is computed on its own: labels, counts and statistics are per chunk and do not join \
across chunk edges, unless the context says the layer is one chunk; say so when it matters. Coarser \
levels have fewer, bigger voxels, so sizes in voxels differ by level. The code runs on every chunk \
on screen, so keep it quick.

Reply with a sentence or two on what the code does, then all of the code in one \`\`\`python block, \
its # lines first. Answer a question that asks for no change without code.`;

/** The providers offered, by Deep Chat's name for its connection: the key each takes (none
 * for Ollama, on this computer), the model the box starts with (shown, so the visitor sees
 * what answers; any other can be typed over it: Claude's others are offered), and a note
 * in place of the one about keys. The models are Deep Chat 2.5.1's defaults, but for
 * Claude's; check them when it is upgraded. */
export const PROVIDERS: Record<string, { name: string; key: string | null; model: string; models?: string[]; note?: string }> = {
  claude: { name: "Claude (Anthropic)", key: "Anthropic API key", model: "claude-opus-5-5", models: ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"] },
  openAI: { name: "OpenAI", key: "OpenAI API key", model: "gpt-5.4" },
  gemini: { name: "Gemini (Google)", key: "Gemini API key", model: "gemini-2.5-flash" },
  openRouter: { name: "OpenRouter (many models)", key: "OpenRouter API key", model: "anthropic/claude-sonnet-5-5" },
  mistral: { name: "Mistral", key: "Mistral API key", model: "mistral-small-latest" },
  deepSeek: { name: "DeepSeek", key: "DeepSeek API key", model: "deepseek-chat" },
  groq: { name: "Groq", key: "Groq API key", model: "llama-3.3-70b-versatile" },
  ollama: { name: "Ollama (this computer, free)", key: null, model: "llama3.2",
    note: "Ollama answers a web page only if started with OLLAMA_ORIGINS set to allow it, e.g. OLLAMA_ORIGINS=\"*\" ollama serve. Name a model you have pulled. The code a reply ends with replaces the code above and runs here." },
};

/** The last fenced block of a reply that defines apply, if one does (whatever the fence
 * says the language is: a reply may have shell or text blocks too). */
export function replyCode(text: string): string | null {
  const blocks = [...text.matchAll(/^[ \t]*```[^\n]*\n([\s\S]*?)^[ \t]*```[ \t]*$/gm)].map((m) => m[1]);
  const code = blocks.reverse().find((b) => /^\s*def apply\s*\(/m.test(b));
  return code ? `${code.trimEnd()}\n` : null;
}

/** `note` before the text of the last message of a request, whichever provider's shape it
 * has: Claude's and OpenAI-style `messages`, OpenAI's `input`, Gemini's `contents`. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function prefixLast(body: any, note: string) {
  if (typeof body?.input === "string") { body.input = note + body.input; return; }
  const last = [body?.messages, body?.input, body?.contents].find(Array.isArray)?.at(-1);
  if (typeof last?.content === "string") { last.content = note + last.content; return; }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const part = [last?.content, last?.parts].find(Array.isArray)?.find((p: any) => typeof p?.text === "string");
  if (part) part.text = note + part.text;
}

export interface ChatHooks {
  /** The layer, its data and its code now, for the next message. */
  context(): string;
  /** Code a reply ends with: into the editor, and run. */
  put(code: string): void;
}

/** The chat in `box`, in place of its `.chat`: the page loads this module (and Deep Chat) the
 * first time the visitor opens it. Deep Chat connects once, when it is first drawn, so a new
 * provider, model or key is a new chat, the conversation carried over. */
export function chatPanel(box: HTMLElement, hooks: ChatHooks) {
  const provider = box.querySelector<HTMLSelectElement>("#ask-provider")!;
  const model = box.querySelector<HTMLInputElement>("#ask-model")!, models = box.querySelector("datalist")!;
  const key = box.querySelector<HTMLInputElement>("#ask-key")!, note = box.querySelector<HTMLElement>("#ask-note")!;
  const said = note.textContent;  // about keys
  const keys: Record<string, string> = {}, picked: Record<string, string> = {};
  for (const [id, p] of Object.entries(PROVIDERS)) provider.append(new Option(p.name, id));
  const connection = (): DirectConnection => {
    const id = provider.value, m = model.value.trim() || PROVIDERS[id].model;
    const settings = { key: key.value.trim() || "none", system_prompt: SYSTEM, model: m };
    if (id === "claude") return { claude: { ...settings, max_tokens: 16000 } };
    if (id === "openAI") return { openAI: { key: settings.key, chat: { instructions: SYSTEM, model: m } } };  // its Responses API
    return { [id]: settings };
  };
  const show = () => {  // the key field and model for the provider picked
    const p = PROVIDERS[provider.value];
    key.hidden = !p.key;
    key.placeholder = p.key ? `Your ${p.key}` : "";
    key.value = keys[provider.value] ?? "";
    models.replaceChildren(...(p.models ?? []).map((m) => new Option(m, m)));
    model.value = picked[provider.value] ?? p.model;
    model.placeholder = p.model;
    note.textContent = p.note ?? said;
  };
  let chat: DeepChat | undefined, timer = 0, waiting = false;  // a request out, its reply to come
  const make = () => {
    if (waiting) return again();  // a new element would never show the reply
    const was = chat;
    chat = document.createElement("deep-chat") as DeepChat;
    const history = (was?.getMessages() ?? []).filter((m) => (m.role === "user" || m.role === "ai") && m.text);
    // replies whole, not streamed: Deep Chat's streams lose an error sent before any text
    Object.assign(chat, { ...LOOK, ...(history.length ? { history } : {}), directConnection: connection(),
      // before each request: the layer's context in the message; Claude's effort, and a
      // request it declines re-run on the model Anthropic recommends
      requestInterceptor: (details: RequestDetails) => {
        const p = PROVIDERS[provider.value];
        if (p.key && !key.value.trim()) return { error: `Paste your ${p.key} above first: it stays in this tab, for this visit.` };
        try {
          prefixLast(details.body, `<context>\n${hooks.context()}\n</context>\n\n`);
        } catch (e) {
          return { error: String((e as Error).message ?? e) };  // the page has no layer to describe yet
        }
        if (provider.value === "claude" && p.models!.includes(details.body.model)) {
          details.body.output_config = { effort: "medium" };
          if (!details.body.model.includes("haiku")) {
            details.body.fallbacks = "default";
            details.headers = { ...details.headers, "anthropic-beta": "server-side-fallback-2026-07-01" };
          }
        }
        waiting = true;
        window.setTimeout(() => { waiting = false; }, 120_000);  // a reply that never comes
        return details;
      },
      onMessage: ({ message, isHistory }: { message: { role?: string; text?: string }; isHistory: boolean }) => {
        if (!isHistory && message.role !== "user") waiting = false;
        const code = !isHistory && message.role === "ai" && message.text ? replyCode(message.text) : null;
        if (code) hooks.put(code);
      },
      onError: () => { waiting = false; },
    });
    (was ?? box.querySelector(".chat")!).replaceWith(chat);
  };
  const again = () => { clearTimeout(timer); timer = window.setTimeout(make, 300); };  // as the key or model is typed
  provider.addEventListener("change", () => { clearTimeout(timer); show(); make(); });
  model.addEventListener("input", () => { picked[provider.value] = model.value.trim(); again(); });
  key.addEventListener("input", () => { keys[provider.value] = key.value.trim(); again(); });
  show();
  make();
}

/** How the chat looks: the page's dark colours. */
const LOOK = {
  errorMessages: { displayServiceErrorMessages: true },
  introMessage: { text: "Say what this layer should show, e.g. \"threshold it with a slider and number the connected pieces\". The code I write replaces the code above and runs." },
  textInput: { placeholder: { text: "What should the code do?" }, styles: { container: { backgroundColor: "#0e1014", border: "1px solid #262c36", color: "#e7e9ec" } } },
  chatStyle: { width: "100%", height: "380px", backgroundColor: "#0e1014", border: "1px solid #262c36", borderRadius: "6px", fontSize: "13px", fontFamily: "system-ui, sans-serif" },
  messageStyles: {
    default: {
      shared: { bubble: { maxWidth: "92%", padding: "6px 9px", lineHeight: "1.45" } },
      ai: { bubble: { backgroundColor: "#161a20", color: "#e7e9ec" } },
      user: { bubble: { backgroundColor: "#7ab0ff", color: "#081225" } },
    },
    intro: { bubble: { backgroundColor: "#161a20", color: "#939cab" } },
    error: { bubble: { backgroundColor: "#2a1414", color: "#ff8a80" } },
  },
  auxiliaryStyle: "pre { white-space: pre-wrap; font-size: 11px; } code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }",
};
