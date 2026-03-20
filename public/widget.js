(function () {
  const BASE_URL = "https://YOUR-RENDER-APP.onrender.com"; // change this

  // Inject styles
  const style = document.createElement("style");
  style.textContent = `
    #cb-bubble { position:fixed; bottom:24px; right:24px; width:52px; height:52px;
      border-radius:50%; background:#4f46e5; color:#fff; font-size:24px; border:none;
      cursor:pointer; display:flex; align-items:center; justify-content:center;
      box-shadow:0 4px 14px rgba(0,0,0,0.25); z-index:9999; }
    #cb-box { position:fixed; bottom:88px; right:24px; width:340px; height:480px;
      background:#fff; border-radius:16px; box-shadow:0 8px 30px rgba(0,0,0,0.15);
      display:none; flex-direction:column; z-index:9999; font-family:sans-serif; overflow:hidden; }
    #cb-header { background:#4f46e5; color:#fff; padding:14px 16px; font-weight:600; font-size:15px; }
    #cb-messages { flex:1; overflow-y:auto; padding:12px; display:flex; flex-direction:column; gap:8px; }
    .cb-msg { max-width:80%; padding:9px 12px; border-radius:12px; font-size:14px; line-height:1.5; word-wrap:break-word; }
    .cb-user { align-self:flex-end; background:#4f46e5; color:#fff; border-bottom-right-radius:4px; }
    .cb-bot { align-self:flex-start; background:#f3f4f6; color:#111; border-bottom-left-radius:4px; }
    #cb-input-row { display:flex; gap:8px; padding:10px; border-top:1px solid #e5e7eb; }
    #cb-input { flex:1; padding:9px 12px; border:1px solid #d1d5db; border-radius:8px;
      font-size:14px; outline:none; }
    #cb-send { background:#4f46e5; color:#fff; border:none; border-radius:8px;
      padding:9px 14px; cursor:pointer; font-size:14px; }
    #cb-send:disabled { opacity:0.5; cursor:not-allowed; }
  `;
  document.head.appendChild(style);

  // Build DOM
  const bubble = document.createElement("button");
  bubble.id = "cb-bubble";
  bubble.innerHTML = "💬";

  const box = document.createElement("div");
  box.id = "cb-box";
  box.innerHTML = `
    <div id="cb-header">Chat with us</div>
    <div id="cb-messages"></div>
    <div id="cb-input-row">
      <input id="cb-input" placeholder="Type a message..." />
      <button id="cb-send">Send</button>
    </div>
  `;

  document.body.appendChild(bubble);
  document.body.appendChild(box);

  const msgs = document.getElementById("cb-messages");
  const input = document.getElementById("cb-input");
  const sendBtn = document.getElementById("cb-send");
  let history = [];
  let open = false;

  bubble.addEventListener("click", () => {
    open = !open;
    box.style.display = open ? "flex" : "none";
    if (open) input.focus();
  });

  function addMsg(text, role) {
    const el = document.createElement("div");
    el.className = "cb-msg " + (role === "user" ? "cb-user" : "cb-bot");
    el.textContent = text;
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
    return el;
  }

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    sendBtn.disabled = true;
    addMsg(text, "user");
    const thinking = addMsg("...", "bot");

    try {
      const res = await fetch(`${BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history }),
      });
      const data = await res.json();
      const reply = data.reply || data.detail || "Something went wrong.";
      thinking.textContent = reply;
      history.push({ role: "user", content: text });
      history.push({ role: "assistant", content: reply });
      if (history.length > 12) history = history.slice(-12);
    } catch {
      thinking.textContent = "Connection error. Please try again.";
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
})();
