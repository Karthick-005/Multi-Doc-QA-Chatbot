document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const fileStatus = document.getElementById("fileStatus");
  const fileList = document.getElementById("fileList");
  const fileTabs = document.getElementById("fileTabs");
  const urlInput = document.getElementById("urlInput");
  const processUrlsBtn = document.getElementById("processUrlsBtn");

  // File Upload Handling
  dropZone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", handleFileUpload);

  ["dragover", "dragleave"].forEach((event) => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.style.background =
        e.type === "dragover" ? "rgba(42, 111, 255, 0.1)" : "transparent";
    });
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length) handleFileUpload({ target: { files } });
  });

  async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      fileStatus.textContent = "Uploading...";
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      if (result.error) throw new Error(result.error);

      fileStatus.textContent = "File processed successfully!";
      fileStatus.style.color = "#4CAF50";
      refreshFileList();
    } catch (error) {
      fileStatus.textContent = `Error: ${error.message}`;
      fileStatus.style.color = "#f44336";
    }
  }

  // URL Processing
  processUrlsBtn.addEventListener("click", async () => {
    console.log("comes here");
    const urls = urlInput.value.split(',').map(url => url.trim()).filter(url => url);
    if (urls.length === 0) {
        alert("Please enter at least one URL.");
        return;
    }

    fileStatus.textContent = "Processing URLs...";
    try {
      const response = await fetch("/process_urls", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls:urls }) // Fix: Convert to JSON string
    });

        const result = await response.json();
        if (result.error) throw new Error(result.error);

        fileStatus.textContent = "URLs processed successfully!";
        refreshFileList(); // Refresh the file list to show new entries
    } catch (error) {
        fileStatus.textContent = `Error: ${error.message}`;
    }
});
  // File List Management
  async function refreshFileList() {
    try {
      const response = await fetch("/list_files");
      const data = await response.json();

      const categorized = data.files.reduce((acc, file) => {
        const ext = file.split(".").pop().toLowerCase();
        acc[ext] = acc[ext] || [];
        acc[ext].push(file);
        return acc;
      }, {});

      const iconMap = {
        pdf: "fa-file-pdf",
        docx: "fa-file-word",
        csv: "fa-file-csv",
        txt: "fa-file-alt",
        html: "fa-file-code",
        wav: "fa-file-audio",
        mp3: "fa-file-audio",
        mp4: "fa-file-video",
        jpg: "fa-file-image",
        png: "fa-file-image",
        jpeg: "fa-file-image"
      };

      // Generate Tabs
      fileTabs.innerHTML = Object.keys(categorized)
        .map(
          (ext) => `
                <button class="tab-btn ${
                  ext === "pdf" ? "active" : ""
                }" data-ext="${ext}">
                    <i class="fas ${iconMap[ext] || "fa-file"}"></i>
                    <span>${ext.toUpperCase()}</span>
                    <div class="file-count">${categorized[ext].length}</div>
                </button>
            `
        )
        .join("");

      // Generate File List
      fileList.innerHTML = Object.entries(categorized)
      .map(([ext, files]) => `
          <div class="file-category ${ext === "pdf" ? "active" : ""}" data-ext="${ext}">
              <h4 class="category-title">${ext.toUpperCase()} Files</h4>
              ${files.map(file => `
                  <div class="file-item" data-file="${file}">
                      <span>${file}</span>
                      <div class="file-actions">
                        <!--  <button class="preview-btn" data-file="${file}">
                              <i class="fas fa-eye"></i>
                          </button>
                          <button class="download-btn" data-file="${file}">
                              <i class="fas fa-download"></i>
                          </button> -->
                          <button class="delete-btn" data-file="${file}">
                              <i class="fas fa-trash"></i>
                          </button>
                      </div>
                  </div>
              `).join("")}
          </div>
      `).join("");

      // Tab Interactions
      document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          document
            .querySelectorAll(".tab-btn, .file-category")
            .forEach((el) => {
              el.classList.remove("active");
            });
          btn.classList.add("active");
          document
            .querySelector(`.file-category[data-ext="${btn.dataset.ext}"]`)
            .classList.add("active");
        });
      });

      // Preview Handlers
      // Inside the refreshFileList function
      document.querySelectorAll(".preview-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const file = btn.dataset.file;
          showPreview(`${file}`);
        });
      });

      // Download Handlers
      document.querySelectorAll(".download-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const file = btn.dataset.file;
          try {
            const response = await fetch(`/download/${file}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = file;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
          } catch (error) {
            console.error("Download failed:", error);
          }
        });
      });
    } catch (error) {
      console.error("Error loading file list:", error);
    }
  }
  function showPreview(fileName) {
    const preview = document.createElement('div');
    preview.className = 'preview-modal';
    
    // Use API endpoint instead of direct file access
    const apiUrl = `/api/file/${encodeURIComponent(fileName)}`;
    
    const ext = fileName.split('.').pop().toLowerCase();
    
    let content = '';
    if (ext === 'pdf') {
      content = `<iframe src="${apiUrl}" class="pdf-iframe"></iframe>`;
    } else if (['png', 'jpg', 'jpeg', 'gif'].includes(ext)) {
      content = `<img src="${apiUrl}" class="preview-image">`;
    } else if (['txt', 'csv'].includes(ext)) {
      fetch(apiUrl)
        .then(response => response.text())
        .then(text => {
          content = `<pre>${escapeHtml(text)}</pre>`;
          updatePreview();
        });
    }
    
    function updatePreview() {
      preview.innerHTML = `
        <div class="preview-header">
          <h3>${fileName}</h3>
          <button class="close-btn">&times;</button>
        </div>
        <div class="preview-content">
          ${content}
        </div>
      `;
      
      // Add close handler
      preview.querySelector('.close-btn').addEventListener('click', () => {
        preview.remove();
      });
    }
    
    updatePreview();
    document.body.appendChild(preview);
  }
  
  // Helper function for HTML escaping
  function escapeHtml(unsafe) {
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
      
  // Initial load
  refreshFileList();
});

// Theme Management
function initializeTheme() {
  const savedTheme = localStorage.getItem("theme") || "dark";
  document.body.setAttribute("data-theme", savedTheme);
}

function toggleTheme() {
  const currentTheme = document.body.getAttribute("data-theme");
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  document.body.setAttribute("data-theme", newTheme);
  localStorage.setItem("theme", newTheme);
}

// Add to DOMContentLoaded event
document.addEventListener("DOMContentLoaded", () => {
  initializeTheme();
  document.getElementById("themeToggle").addEventListener("click", toggleTheme);
});

document.addEventListener("click", function (event) {
  if (event.target.closest(".delete-btn")) {
      const fileItem = event.target.closest(".file-item");
      const fileName = fileItem.getAttribute("data-file");
      const categoryDiv = fileItem.closest(".file-category");
      const ext = categoryDiv.getAttribute("data-ext");
      const tabButton = document.querySelector(`.tab-btn[data-ext="${ext}"]`);

      if (confirm(`Are you sure you want to delete ${fileName}?`)) {
          fetch(`/delete-file?filename=${encodeURIComponent(fileName)}`, {
              method: "DELETE",
          })
          .then(response => response.json())
          .then(data => {
              if (data.success) {
                  fileItem.remove();
                  console.log(`Deleted: ${fileName}`);

                  // Check if the category is now empty
                  if (categoryDiv.querySelectorAll(".file-item").length === 0) {
                      categoryDiv.remove(); // Remove the empty category
                      if (tabButton) {
                          tabButton.remove(); // Remove the corresponding tab
                      }
                  }
              } else {
                  console.error("Error deleting file:", data.error);
              }
          })
          .catch(error => console.error("Request failed:", error));
      }
  }
});
