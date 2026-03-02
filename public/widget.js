(function() {
  const bubble = document.createElement("div");
  bubble.innerHTML = "💬";
  bubble.style.position = "fixed";
  bubble.style.bottom = "20px";
  bubble.style.right = "20px";
  bubble.style.cursor = "pointer";
  bubble.style.fontSize = "30px";
  document.body.appendChild(bubble);

  const chatBox = document.createElement("div");
  chatBox.style.position = "fixed";
  chatBox.style.bottom = "70px";
  chatBox.style.right = "20px";
  chatBox.style.width = "300px";
  chatBox.style.height = "400px";
  chatBox.style.background = "#fff";
  chatBox.style.border = "1px solid #ccc";
  chatBox.style.display = "none";
  chatBox.style.flexDirection = "column";
  chatBox.style.padding = "10px";
  document.body.appendChild(chatBox);

  const messages = document.createElement("div");
  messages.style.flex = "1";
  messages.style.overflowY = "auto";
  chatBox.appendChild(messages);

  const input = document.createElement("input");
  input.placeholder = "Ask something...";
  chatBox.appendChild(input);

  bubble.onclick = () => {
    chatBox.style.display = chatBox.style.display === "none" ? "flex" : "none";
  };

  input.addEventListener("keypress", async (e) => {
    if (e.key === "Enter") {
      const question = input.value;
      input.value = "";

      try {
        const res = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query: question})
        });

        const data = await res.json();
        messages.innerHTML += "<div><b>You:</b> " + question + "</div>";
        messages.innerHTML += "<div><b>Bot:</b> " + data.answer + "</div>";
      } catch (err) {
        messages.innerHTML += "<div>Error connecting to server.</div>";
      }
    }
  });
})();