// The gallery: one card per demo in cards.ts.
import { CARDS } from "./cards";

const root = document.getElementById("cards")!;
for (const c of CARDS) {
  const href = c.kind === "pipeline" ? `pipeline.html?card=${c.id}` : c.kind === "link" ? c.href : null;
  const card = document.createElement("article");
  card.className = "card";
  card.innerHTML = `
    <a href=""><img alt="" loading="lazy"></a>
    <div class="body">
      <h2></h2>
      <p class="blurb"></p>
      <p class="data"></p>
      <details><summary>The same from Python</summary><pre></pre></details>
      <div class="actions"><a class="open" href="">Open</a></div>
    </div>`;
  if (href) for (const a of card.querySelectorAll("a")) a.href = href;
  else {  // Python only: the command is the way in
    card.querySelector("img")!.parentElement!.replaceWith(card.querySelector("img")!);
    card.querySelector("details")!.open = true;
    card.querySelector("summary")!.textContent = "Run it from Python";
    card.querySelector(".actions")!.innerHTML = `<p class="why"></p>`;
    card.querySelector(".why")!.textContent = (c as { why?: string }).why ?? "";
  }
  card.querySelector("img")!.src = c.image;
  card.querySelector("h2")!.textContent = c.title;
  card.querySelector(".blurb")!.textContent = c.blurb;
  card.querySelector(".data")!.textContent = `Data: ${c.data}`;
  card.querySelector("pre")!.textContent = c.command;
  root.append(card);
}
