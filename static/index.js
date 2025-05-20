document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileStatus = document.getElementById('fileStatus');
    const fileList = document.getElementById('fileList');
    const fileTabs = document.getElementById('fileTabs');

    // File Upload Handling
    dropZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', handleFileUpload);
    
    ['dragover', 'dragleave'].forEach(event => {
        dropZone.addEventListener(event, e => {
            e.preventDefault();
            dropZone.style.background = e.type === 'dragover' ? 
                'rgba(42, 111, 255, 0.1)' : 'transparent';
        });
    });

    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length) handleFileUpload({ target: { files } });
    });

    async function handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            fileStatus.textContent = 'Uploading...';
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (result.error) throw new Error(result.error);

            fileStatus.textContent = 'File processed successfully!';
            fileStatus.style.color = 'var(--success-color)';
            refreshFileList();
        } catch (error) {
            fileStatus.textContent = `Error: ${error.message}`;
            fileStatus.style.color = 'var(--error-color)';
        }
    }

    // File List Management
    async function refreshFileList() {
        try {
            const response = await fetch('/list_files');
            const data = await response.json();
            
            // Categorize files
            const categorized = data.files.reduce((acc, file) => {
                const ext = file.split('.').pop().toLowerCase();
                acc[ext] = acc[ext] || [];
                acc[ext].push(file);
                return acc;
            }, {});

            // Generate tabs
            const iconMap = {
                pdf: 'fa-file-pdf',
                docx: 'fa-file-word',
                csv: 'fa-file-csv',
                txt: 'fa-file-alt',
                xlsx: 'fa-file-excel',
                pptx: 'fa-file-powerpoint'
            };

            const tabsHTML = Object.keys(categorized).map(ext => `
                <button class="tab-btn ${ext === 'pdf' ? 'active' : ''}" data-ext="${ext}">
                    <i class="fas ${iconMap[ext] || 'fa-file'}"></i>
                    <span>${ext.toUpperCase()}</span>
                    <div class="file-count">${categorized[ext].length}</div>
                </button>
            `).join('');

            fileTabs.innerHTML = tabsHTML;

            // Generate file list
            const filesHTML = Object.entries(categorized).map(([ext, files]) => `
                <div class="file-category ${ext === 'pdf' ? 'active' : ''}" data-ext="${ext}">
                    <h4 class="category-title">${ext.toUpperCase()} Files</h4>
                    ${files.map(file => `
                        <div class="file-item">
                            <span>${file}</span>
                            <div class="file-actions">
                                <button class="preview-btn" data-file="${file}">
                                    <i class="fas fa-eye"></i> Preview
                                </button>
                                <button class="download-btn" data-file="${file}">
                                    <i class="fas fa-download"></i>
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `).join('');

            fileList.innerHTML = filesHTML;

            // Add tab click handlers
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.tab-btn, .file-category').forEach(el => {
                        el.classList.remove('active');
                    });
                    btn.classList.add('active');
                    document.querySelector(`.file-category[data-ext="${btn.dataset.ext}"]`)
                        .classList.add('active');
                });
            });

            // Add preview handlers
            document.querySelectorAll('.preview-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const file = btn.dataset.file;
                    const response = await fetch(`/preview/${file}`);
                    const data = await response.json();
                    showPreview(data.content, file);
                });
            });

            // Add download handlers
            document.querySelectorAll('.download-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const file = btn.dataset.file;
                    try {
                        const response = await fetch(`/download/${file}`);
                        if (response.ok) {
                            const blob = await response.blob();
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = file;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            window.URL.revokeObjectURL(url);
                        }
                    } catch (error) {
                        console.error('Download failed:', error);
                    }
                });
            });

        } catch (error) {
            console.error('Error loading file list:', error);
        }
    }

    function showPreview(content, fileName) {
        const ext = fileName.split('.').pop().toLowerCase();
        const preview = document.createElement('div');
        preview.className = 'preview-modal';
      
        let contentDisplay = '';
      
        if (['txt', 'csv', 'log','pdf','docx'].includes(ext)) {
          contentDisplay = `<p>${content.substring(0, 5000)}...</p>`;
        } else if (['jpg', 'jpeg', 'png', 'gif'].includes(ext)) {
          contentDisplay = `<img src="data:image/${ext};base64,${content}" alt="Preview" style="max-width: 100%; max-height: 80vh;">`;
        } 
        else {
          contentDisplay = `<p>Preview not supported for this file type.</p>`;
        }
      
        preview.innerHTML = `
          <div class="preview-content">
            <h3>File Preview: ${fileName}</h3>
            ${contentDisplay}
            <button class="close-btn">Close</button>
          </div>
        `;
      
        document.body.appendChild(preview);
      
        preview.querySelector('.close-btn').addEventListener('click', () => {
          preview.remove();
        });
    }

    // Initial load of file list
    refreshFileList();
});