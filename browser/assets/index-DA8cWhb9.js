import"./modulepreload-polyfill-P2Xu9kJm.js";import{r as e,t}from"./cards--9-MEbNz.js";var n=document.getElementById(`cards`);for(let r of t){let t=r.kind===`pipeline`?`pipeline.html?card=${r.id}`:r.kind===`map`?`map.html?card=${r.id}`:r.kind===`link`?r.href:null,i=document.createElement(`article`);if(i.className=`card`,i.innerHTML=`
    <a href=""><img alt="" loading="lazy"></a>
    <div class="body">
      <h2></h2>
      <p class="blurb"></p>
      <p class="data"></p>
      <details><summary>The same from Python</summary><pre></pre></details>
      <div class="actions"><a class="open" href="">Open</a></div>
    </div>`,t)for(let e of i.querySelectorAll(`a`))e.href=t;else i.querySelector(`img`).parentElement.replaceWith(i.querySelector(`img`)),i.querySelector(`details`).open=!0,i.querySelector(`summary`).textContent=`Run it from Python`,i.querySelector(`.actions`).innerHTML=`<p class="why"></p>`,i.querySelector(`.why`).textContent=r.why??``;if(i.querySelector(`img`).src=r.image,i.querySelector(`h2`).textContent=r.title,i.querySelector(`.blurb`).textContent=r.blurb,i.querySelector(`.data`).textContent=`Data: ${r.data}`,i.querySelector(`pre`).textContent=r.command,e(r).length){let e=document.createElement(`span`);e.className=`tag`,e.textContent=`✎ Your Python`,e.title=`Its page has a panel for Python of your own, run on a layer's chunks`,i.querySelector(`.actions`).append(e)}n.append(i)}