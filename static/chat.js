let chatHistoryData = JSON.parse(localStorage.getItem('chatHistory')) || [];
let currentChatId = Date.now();
let currentChat = [];

window.addEventListener('load', () => {
  // Add titles and files to existing chats for backward compatibility
  chatHistoryData = chatHistoryData.map(chat => {
    const updatedMessages = chat.messages.map(message => ({
      ...message,
      file: message.file || "default.pdf"
    }));
    
    return {
      ...chat,
      title: chat.title || (chat.messages[0]?.question || "New Chat"),
      messages: updatedMessages
    };
  });
  localStorage.setItem('chatHistory', JSON.stringify(chatHistoryData));

  updateSidebar();
  
  // New code: Always start with fresh chat
  document.getElementById("welcomeMessage").style.display = 'block';
  currentChatId = Date.now();
  currentChat = [];
});

function goToUpload() {
  saveCurrentChat();
  window.location.href = "/";
}

function toggleSelection() {
  const docSelection = document.getElementById("docSelection");
  docSelection.classList.toggle("hidden");
}

function toggleSidebar() {
  const sidebar = document.getElementById("chatHistorySidebar");
  sidebar.classList.toggle("show");
}

function saveCurrentChat() {
  if (currentChat.length > 0) {
    const existingIndex = chatHistoryData.findIndex(chat => chat.id === currentChatId);
    const chatData = {
      id: currentChatId,
      messages: [...currentChat],
      timestamp: new Date().toLocaleString(),
      title: currentChat[0].question || "New Chat"
    };

    if (existingIndex > -1) {
      chatHistoryData[existingIndex] = chatData;
    } else {
      chatHistoryData.unshift(chatData);
    }

    localStorage.setItem('chatHistory', JSON.stringify(chatHistoryData));
    updateSidebar();
  }
}

function updateSidebar() {
  const sidebarChatHistory = document.getElementById("sidebarChatHistory");
  sidebarChatHistory.innerHTML = '';

  chatHistoryData.sort((a, b) => b.id - a.id).forEach(chat => {
    const listItem = document.createElement("li");
    listItem.className = "sidebar-chat-item";
    listItem.innerHTML = `
            <div class="chat-info">
                <div class="chat-title">${chat.title}</div>
                <button class="delete-chat-btn"><img src="/static/remove.png"></button>
            </div>
        `;

    listItem.querySelector('.chat-info').addEventListener("click", () => {
      saveCurrentChat();
      loadChat(chat.id);
      toggleSidebar();
    });

    listItem.querySelector('.delete-chat-btn').addEventListener("click", (e) => {
      e.stopPropagation();
      if (confirm("Are you sure you want to delete this chat?")) {
        chatHistoryData = chatHistoryData.filter(c => c.id !== chat.id);
        localStorage.setItem('chatHistory', JSON.stringify(chatHistoryData));
        // Remove the DOM element immediately
        listItem.remove();  // <-- Add this line
        if (chat.id === currentChatId) {
          currentChat = [];
          currentChatId = Date.now();
          document.getElementById("chatHistory").innerHTML = '';
          document.getElementById("welcomeMessage").style.display = 'block';
        }

        updateSidebar();
      }
    });

    sidebarChatHistory.appendChild(listItem);
  });
}

function loadChat(chatId) {
  saveCurrentChat();

  const chatHistory = document.getElementById("chatHistory");
  const welcomeMessage = document.getElementById("welcomeMessage");

  chatHistory.innerHTML = '';
  if (welcomeMessage) welcomeMessage.style.display = 'none';

  const chatSession = chatHistoryData.find(chat => chat.id === chatId);
  if (chatSession) {
    chatSession.messages.forEach(message => {
      const userDiv = document.createElement("div");
      userDiv.className = "chat-message user-message";
      userDiv.textContent = message.question;
      chatHistory.appendChild(userDiv);

      const botDiv = document.createElement("div");
      botDiv.className = "chat-message bot-message";
      botDiv.innerHTML = message.answer;

      const contextButton = document.createElement("button");
      contextButton.textContent = "Relevant Context >>";
      contextButton.className = "context-btn";
      contextButton.onclick = () => getContext(message.question, message.file);

      botDiv.appendChild(contextButton);
      chatHistory.appendChild(botDiv);
    });

    currentChatId = chatId;
    currentChat = [...chatSession.messages];
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
}

function truncateText(text, wordLimit = 50) {
  const words = text.split(/\s+/); // Split by whitespace
  if (words.length > wordLimit) {
    return words.slice(0, wordLimit).join(" ") + " ...";
  }
  return text;
}

let globalContexts = []; // Global variable to hold contexts

function showContext(contexts) {
    globalContexts = contexts; // Store contexts in the global variable
    const contextContainer = document.getElementById("contextContainer");
    const contextContent = document.getElementById("contextContent");

    // Generate HTML for all context chunks
    const chunksHtml = Array.isArray(contexts) && contexts.length > 0 
    ? contexts.map((context, index) => {
        return `
            <div class="context-chunk">
                <div class="chunk-content">
                    ${truncateText(context.page_content)}
                </div>
                <div class="chunk-footer">
                    <a href="#" 
                       class="context-link"
                       data-chunk-id="${index}"
                       onclick="openModal(event)">
                       Open 
                    </a>
                </div>
            </div>
        `;
    }).join('')
    : `<div class="no-context">No context found</div>`;

    contextContent.innerHTML = `
        <div class="context-header">
            <h3>Relevant Context (${contexts.length} chunks found)</h3>
            <button class="close-context-btn" onclick="closeContext()">Close</button>
        </div>
        <div class="contexts-list">
            ${chunksHtml}
        </div>
    `;

    contextContainer.classList.add("show");
}

function openModal(event) {
  event.preventDefault(); // Prevent default anchor click behavior
  const chunkId = event.target.getAttribute('data-chunk-id');
  const selectedContext = globalContexts[chunkId]; // Access the global contexts

  // Populate the modal with the selected context content
  const modalBody = document.getElementById("modal-body");
  modalBody.innerHTML = `
      <h4 class='popup-title'>Context Details</h4>
      <p>${selectedContext.page_content}</p>
  `;

  // Show the modal
  const modal = document.getElementById("modal");
  modal.style.display = "block";
  const blurOverlay = document.getElementById("blur-overlay");
  blurOverlay.style.display = "block";
}

function closeModal() {
  const modal = document.getElementById("modal");
  modal.style.display = "none";
  const blurOverlay = document.getElementById("blur-overlay");
  blurOverlay.style.display = "none";

}

window.onclick = function(event) {
  const modal = document.getElementById("modal");
  const blurOverlay = document.getElementById("blur-overlay");
  if (event.target === modal || event.target === blurOverlay) {
      closeModal();
  }
}

function closeContext() { 
  const contextContainer = document.getElementById("contextContainer"); 
  contextContainer.classList.remove("show"); 
}

function getContext(question, filename) {
  fetch("/get_context", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, file: filename })
  })
  .then(response => {
    if (!response.ok) {
      return response.json().then(err => { throw new Error(err.error || 'Unknown error') });
    }
    return response.json();
  })
  .then(data => {
    showContext(data.contexts);  // Now only passing contexts
  })
  .catch(error => {
    console.error("Error fetching context:", error);
    showContext([{ 
      page_content: error.message || "Error loading context. Please try again.",
      metadata: { page_no: 0, filename: '' }
    }]);
  });
}

document.addEventListener("DOMContentLoaded", function () {
  const questionInput = document.getElementById("questionInput");
  const askButton = document.getElementById("askButton");
  const chatHistory = document.getElementById("chatHistory");
  const dropdownBtn = document.getElementById("dropdownBtn");
  const dropdownList = document.getElementById("dropdownList");
  const welcomeMessage = document.getElementById("welcomeMessage");
  const newChatButton = document.getElementById("newChatButton");

  function askQuestion() {
    const question = questionInput.value.trim();
    const selectedFile = dropdownBtn.textContent.trim();

    if (!selectedFile || selectedFile === "Select a document") {
        alert("Please select a file before asking a question.");
        return;
    }

    if (!question) {
        alert("Please enter a question.");
        return;
    }
    welcomeMessage.style.display = 'none';

    // Create user message with logo
    const userDiv = document.createElement("div");
    userDiv.className = "chat-message user-message";
    userDiv.innerHTML = `
        <div class="message-content">
            <span>${question}</span>
        </div>
    `;
    chatHistory.appendChild(userDiv);

    // Create loading message
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "chat-message bot-message loader-container";

    const loader = document.createElement("div");
    loader.className = "loader";

    const thinkingText = document.createElement("div");
    thinkingText.className = "thinking-text";
    thinkingText.textContent = "Thinking...";

    loadingDiv.appendChild(loader);
    loadingDiv.appendChild(thinkingText);
    chatHistory.appendChild(loadingDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;


    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, file: selectedFile })
    })
    .then(response => response.json())
    .then(data => {
        chatHistory.removeChild(loadingDiv);

        // Create bot message with logo
        const botDiv = document.createElement("div");
        document.querySelectorAll('.stop-voice-btn, .play-voice-btn').forEach(btn => btn.style.display = 'none');
        botDiv.className = "chat-message bot-message";
        botDiv.innerHTML = `
            <div class="message-content">
                <span>${data.answer || "Error processing question."}</span>
            </div>
        `;
        speakMessage(data.answer || "Error processing question.");

        const contextButton = document.createElement("button");
        contextButton.className = "context-btn";
        contextButton.textContent = "Relevant Context >>";
        contextButton.style.cssText = `
            background: none;
            cursor: pointer;
            border: none;
            font-size: 15px;
            font-family: cursive;
            margin-top: 15px;
            text-transform: uppercase;
        `;
        contextButton.addEventListener("click", () => getContext(question, selectedFile));

        botDiv.appendChild(contextButton);
        chatHistory.appendChild(botDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        currentChat.push({
            question,
            answer: data.answer,
            file: selectedFile
        });
    })
    .catch(error => {
        console.error("Error:", error);
        chatHistory.removeChild(loadingDiv);
        const errorDiv = document.createElement("div");
        errorDiv.className = "chat-message bot-message";
        errorDiv.style.color = "red";
        errorDiv.textContent = "Error: Failed to get response.";
        chatHistory.appendChild(errorDiv);
    });

    questionInput.value = "";
}

  newChatButton.addEventListener("click", () => {
    saveCurrentChat();
    chatHistory.innerHTML = '';
    welcomeMessage.style.display = 'block';
    currentChatId = Date.now();
    currentChat = [];
  });

  askButton.addEventListener("click", askQuestion);
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") askQuestion();
  });

  function loadFileList() {
    fetch("/list_files")
      .then(response => response.json())
      .then(data => {
        dropdownList.innerHTML = '';
        data.files.forEach(file => {
          const listItem = document.createElement("li");
          listItem.textContent = file;
          listItem.className = "dropdown-item";
          listItem.addEventListener("click", () => {
            dropdownBtn.textContent = file;
            dropdownList.classList.add("hidden");
          });
          dropdownList.appendChild(listItem);
        });
      })
      .catch(error => console.error("Error loading files:", error));
  }

  dropdownBtn.addEventListener("click", () => {
    dropdownList.classList.toggle("hidden");
  });

  document.addEventListener("click", (e) => {
    if (!dropdownBtn.contains(e.target) && !dropdownList.contains(e.target)) {
      dropdownList.classList.add("hidden");
    }
  });

  loadFileList();
});

document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.querySelector('.theme-toggle');
    const body = document.body;

    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        body.classList.toggle('dark-mode', savedTheme === 'dark');
    }

    themeToggle.addEventListener('click', function() {
        body.classList.toggle('dark-mode');
        const isDarkMode = body.classList.contains('dark-mode');
        localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
    });
});


// Voice input functionality with animation
const voiceInputButton = document.getElementById("voice-input");
const msgInput = document.getElementById("questionInput");
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = "en-US";
let isRecording = false;

voiceInputButton.addEventListener("click", () => {
    const icon = voiceInputButton.querySelector('i');
    if (!isRecording) {
        recognition.start();
        isRecording = true;
        icon.classList.replace('fa-microphone', 'fa-stop');
        voiceInputButton.classList.add("recording");
    } else {
        recognition.stop();
        isRecording = false;
        icon.classList.replace('fa-stop', 'fa-microphone');
        voiceInputButton.classList.remove("recording");
    }
});

recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    msgInput.value = transcript;
};

const handleRecognitionEnd = () => {
    const icon = voiceInputButton.querySelector('i');
    isRecording = false;
    icon.classList.replace('fa-stop', 'fa-microphone');
    voiceInputButton.classList.remove("recording");
};

recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    handleRecognitionEnd();
};

recognition.onend = () => {
    handleRecognitionEnd();
};
// Voice output functionality
let currentUtterance = null;
let lastSpokenMessage = null;

function speakMessage(message) {
    if (currentUtterance) {
        window.speechSynthesis.cancel();
    }
    currentUtterance = new SpeechSynthesisUtterance(message);
    currentUtterance.lang = "en-US";
    lastSpokenMessage = message;
    window.speechSynthesis.speak(currentUtterance);
}
