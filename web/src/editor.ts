// The code panel's editor: CodeMirror over a textarea, which stays what the page reads and
// writes (its value, its input events). Python highlighted in VS Code's dark colours, line
// numbers (errors name lines), indenting as you type, and Ctrl+/ (⌘/) commenting lines out
// and in, as in VS Code.
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { bracketMatching, HighlightStyle, indentOnInput, syntaxHighlighting } from "@codemirror/language";
import { EditorState } from "@codemirror/state";
import { drawSelection, EditorView, highlightActiveLine, highlightActiveLineGutter, keymap, lineNumbers } from "@codemirror/view";
import { tags as t } from "@lezer/highlight";

/** VS Code's Dark+ colours for Python. */
const DARK_PLUS = HighlightStyle.define([
  { tag: [t.controlKeyword, t.moduleKeyword], color: "#c586c0" },  // if, return, import
  { tag: [t.keyword, t.definitionKeyword, t.operatorKeyword, t.bool, t.null, t.self], color: "#569cd6" },  // def, and, True
  { tag: [t.function(t.variableName), t.function(t.definition(t.variableName)), t.function(t.propertyName)], color: "#dcdcaa" },
  { tag: [t.variableName, t.definition(t.variableName), t.propertyName], color: "#9cdcfe" },
  { tag: t.className, color: "#4ec9b0" },
  { tag: [t.string, t.special(t.string), t.docString], color: "#ce9178" },
  { tag: t.number, color: "#b5cea8" },
  { tag: [t.comment, t.lineComment], color: "#6a9955" },
]);

const THEME = EditorView.theme({
  "&": { backgroundColor: "#0e1014", color: "#d4d4d4", border: "1px solid #262c36", borderRadius: "6px", fontSize: "12px", maxHeight: "60vh" },
  "&.cm-focused": { outline: "1px solid #7ab0ff" },
  ".cm-scroller": { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", lineHeight: "1.5", overflow: "auto" },
  ".cm-content, .cm-gutter": { minHeight: "16em" },
  ".cm-content": { caretColor: "#aeafad" },
  ".cm-cursor": { borderLeftColor: "#aeafad" },
  ".cm-gutters": { backgroundColor: "#0e1014", color: "#5d6675", border: "none" },
  ".cm-activeLine, .cm-activeLineGutter": { backgroundColor: "#ffffff08" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": { backgroundColor: "#264f78" },
  ".cm-matchingBracket": { backgroundColor: "#ffffff1a", outline: "1px solid #888" },
}, { dark: true });

/** CodeMirror in place of `text`, keeping it up to date (its value, an input event per
 * edit); `set` puts a new value in both, without the input event. */
export function codeEditor(text: HTMLTextAreaElement): { set(value: string): void; view: EditorView } {
  let setting = false;
  const view = new EditorView({
    state: EditorState.create({
      doc: text.value,
      extensions: [
        lineNumbers(), highlightActiveLineGutter(), highlightActiveLine(), drawSelection(), history(),
        indentOnInput(), bracketMatching(), python(), syntaxHighlighting(DARK_PLUS), THEME,
        EditorState.tabSize.of(4),
        keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),  // Mod-/ toggles comments
        EditorView.updateListener.of((u) => {
          if (!u.docChanged || setting) return;
          text.value = u.state.doc.toString();
          text.dispatchEvent(new Event("input"));
        }),
      ],
    }),
  });
  text.hidden = true;
  text.after(view.dom);
  return {
    view,
    set(value: string) {
      text.value = value;
      setting = true;
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: value } });
      setting = false;
    },
  };
}
