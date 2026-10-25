document.addEventListener("DOMContentLoaded", () => {
    const urlForm = document.getElementById("url-form");
    const urlInput = document.getElementById("youtube-url");
    const btnPasteUrl = document.getElementById("btn-paste-url");
    const btnValidateUrl = document.getElementById("btn-validate-url");
    const validateSpinner = document.getElementById("validate-spinner");
    const validateText = btnValidateUrl.querySelector(".btn-validate-text");

    const videoPreviewCard = document.getElementById("video-preview-card");
    const previewThumbnail = document.getElementById("preview-thumbnail");
    const previewTitle = document.getElementById("preview-title");
    const previewChannel = document.getElementById("preview-channel");
    const previewDuration = document.getElementById("preview-duration");

    const clipCountSelect = document.getElementById("clip-count");
    const subtitleStyleSelect = document.getElementById("subtitle-style");
    const transitionStyleSelect = document.getElementById("transition-style");

    const generateButton = document.getElementById("generate-button");
    const btnSpinner = document.getElementById("btn-spinner");
    const btnText = generateButton.querySelector(".btn-text");

    const progressContainer = document.getElementById("progress-container");
    const stageLabel = document.getElementById("stage-label");
    const progressPercentage = document.getElementById("progress-percentage");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const progressInfo = document.getElementById("progress-info");
    const statusMessage = document.getElementById("status-message");

    const compilationSection = document.getElementById("compilation-section");
    const compilationVideo = document.getElementById("compilation-video");
    const btnDownloadCompilation = document.getElementById("btn-download-compilation");

    const emptyState = document.getElementById("empty-state");
    const clipsContainer = document.getElementById("clips-container");
    const clipsCounter = document.getElementById("clips-counter");
    const clipsCountVal = document.getElementById("clips-count-val");

    let pollInterval = null;
    let validatedUrl = "";
    let debounceTimer = null;
    let currentJobId = null;

    function normalizeUrl(raw) {
        if (!raw) return "";
        let clean = raw.trim().replace(/^["']+|["']+$/g, "");
        if (!clean.startsWith("http://") && !clean.startsWith("https://")) {
            clean = "https://" + clean;
        }
        return clean;
    }

    // Clipboard Paste Support
    btnPasteUrl.addEventListener("click", async () => {
        try {
            if (navigator.clipboard && navigator.clipboard.readText) {
                const text = await navigator.clipboard.readText();
                if (text) {
                    const clean = normalizeUrl(text);
                    urlInput.value = clean;
                    validateYouTubeUrl(clean);
                }
            } else {
                urlInput.focus();
                showStatus("Please press Ctrl+V to paste your YouTube URL directly into the field.", false);
            }
        } catch (err) {
            urlInput.focus();
            showStatus("Clipboard access was not permitted. Please press Ctrl+V to paste.", false);
        }
    });

    // Inspect/Validate Button Action
    btnValidateUrl.addEventListener("click", () => {
        const raw = urlInput.value.trim();
        if (raw) {
            const clean = normalizeUrl(raw);
            urlInput.value = clean;
            validateYouTubeUrl(clean);
        } else {
            showStatus("Please paste a YouTube video URL first.", true);
            urlInput.focus();
        }
    });

    // Auto-inspect on typing/pasting with 500ms debounce
    urlInput.addEventListener("input", () => {
        clearTimeout(debounceTimer);
        const val = urlInput.value.trim();
        if (val.length >= 12 && (val.includes("youtube.com") || val.includes("youtu.be"))) {
            debounceTimer = setTimeout(() => {
                const normalized = normalizeUrl(val);
                if (normalized !== validatedUrl) {
                    validateYouTubeUrl(normalized);
                }
            }, 500);
        }
    });

    async function validateYouTubeUrl(url) {
        if (!url) return;
        showStatus("");
        setValidating(true);

        try {
            const response = await fetch("/api/validate-url", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Invalid YouTube video link.");
            }

            // Populate preview card
            previewThumbnail.src = data.thumbnail || "";
            previewTitle.textContent = data.title || "YouTube Video";
            previewChannel.textContent = data.channel || "YouTube";
            previewDuration.textContent = `⏱ ${formatSeconds(data.duration || 0)}`;

            videoPreviewCard.classList.remove("hidden");
            validatedUrl = url;
            showStatus("");
        } catch (error) {
            videoPreviewCard.classList.add("hidden");
            validatedUrl = "";
            showStatus(error.message, true);
        } finally {
            setValidating(false);
        }
    }

    // Form Submission & Pipeline Execution
    urlForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const rawUrl = urlInput.value.trim();
        if (!rawUrl) {
            showStatus("Please paste a supported YouTube video URL.", true);
            urlInput.focus();
            return;
        }

        const url = normalizeUrl(rawUrl);
        urlInput.value = url;

        const payload = {
            url: url,
            clip_count: parseInt(clipCountSelect.value, 10),
            subtitle_style: subtitleStyleSelect.value,
            transition_style: transitionStyleSelect.value
        };

        setLoading(true);
        progressContainer.classList.remove("hidden");
        updateProgress(5, "Connecting to YouTube stream...", "Verifying URL and inspecting formats...");
        showStatus("");

        try {
            const response = await fetch("/api/process-url", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || "Failed to start processing.");
            }

            const jobId = result.job_id;
            currentJobId = jobId;
            updateProgress(10, "Job assigned to worker...", "Starting high-quality stream retrieval...");
            startPolling(jobId);

        } catch (error) {
            setLoading(false);
            progressContainer.classList.add("hidden");
            showStatus(error.message, true);
        }
    });

    function startPolling(jobId) {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/jobs/${jobId}`);
                if (!res.ok) {
                    throw new Error("Failed to check job progress.");
                }

                const job = await res.json();
                updateProgress(job.progress, job.current_stage, getStageDescription(job));

                if (job.status === "completed") {
                    clearInterval(pollInterval);
                    setLoading(false);
                    showStatus("All clips and transitions generated successfully!", false);
                    renderClips(job.clips || [], job.compilation);
                } else if (job.status === "failed") {
                    clearInterval(pollInterval);
                    setLoading(false);
                    stageLabel.textContent = "Processing failed";
                    progressInfo.textContent = job.error || "An error occurred during processing.";
                    showStatus(job.error || "Processing failed.", true);
                }
            } catch (err) {
                clearInterval(pollInterval);
                setLoading(false);
                showStatus(err.message, true);
            }
        }, 1500);
    }

    function getStageDescription(job) {
        if (job.status === "failed") {
            return job.error || "Processing failed.";
        }
        if (job.progress < 25) return "Retrieving source video at original quality...";
        if (job.progress < 35) return "Extracting audio channel and decoding properties...";
        if (job.progress < 50) return "Running AI Whisper speech recognition model...";
        if (job.progress < 60) return "Selecting engaging highlights and pacing windows...";
        if (job.progress < 90) return "Detecting faces, applying smooth framing, and burning captions...";
        if (job.progress < 98) return "Assembling compilation reel with transitions...";
        return "Finalizing vertical Shorts...";
    }

    function updateProgress(percent, stageText, infoText) {
        progressBarFill.style.width = `${percent}%`;
        progressPercentage.textContent = `${percent}%`;
        stageLabel.textContent = stageText;
        if (infoText) progressInfo.textContent = infoText;
    }

    function renderClips(clips, compilation) {
        // Render compilation reel if available
        if (compilation && compilation.video_url) {
            compilationVideo.querySelector("source").src = compilation.video_url;
            compilationVideo.load();
            btnDownloadCompilation.href = compilation.video_url;
            btnDownloadCompilation.download = `clipforge_compilation_${compilation.clip_count}clips.mp4`;
            compilationSection.classList.remove("hidden");
        } else {
            compilationSection.classList.add("hidden");
        }

        if (!clips || clips.length === 0) {
            emptyState.classList.remove("hidden");
            clipsContainer.classList.add("hidden");
            clipsCounter.classList.add("hidden");
            return;
        }

        emptyState.classList.add("hidden");
        clipsContainer.classList.remove("hidden");
        clipsCounter.classList.remove("hidden");
        clipsCountVal.textContent = clips.length;

        clipsContainer.innerHTML = "";

        clips.forEach(clip => {
            const card = document.createElement("div");
            card.className = "clip-card";

            const startFmt = formatSeconds(clip.start);
            const endFmt = formatSeconds(clip.end);

            const isTwoView = clip.layout_mode === "two_view";
            const modeBadgeClass = isTwoView ? "mode-badge two-view" : "mode-badge single";
            const modeBadgeText = isTwoView ? "Mode B: Stacked 2-View" : "Mode A: Single Subject";
            const clipTitle = clip.title || `MrBeast's Insane Challenge #${clip.clip_number}`;

            card.innerHTML = `
                <div class="clip-title-panel">
                    <div class="clip-title-header">
                        <div class="clip-title-meta">
                            <span class="clip-title-label">⚡ YouTube Shorts Title</span>
                            <span class="mrbeast-badge">🔥 MrBeast</span>
                        </div>
                        <button type="button" class="btn-copy-title" data-clip-id="${clip.clip_id}" title="Copy title to clipboard">
                            <span class="copy-icon">📋</span>
                            <span class="copy-text">Copy Title</span>
                        </button>
                    </div>
                    <div class="clip-title-input-wrapper">
                        <input
                            type="text"
                            class="clip-title-input"
                            id="input-title-${clip.clip_id}"
                            data-clip-id="${clip.clip_id}"
                            value="${escapeHtml(clipTitle)}"
                            spellcheck="false"
                            title="Click to edit title before copying"
                            placeholder="Enter title..."
                        />
                    </div>
                </div>
                <div class="video-wrapper">
                    <video controls playsinline preload="metadata" poster="${clip.thumbnail_url}">
                        <source src="${clip.video_url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <div class="${modeBadgeClass}" title="${escapeHtml(clip.layout_description || '')}">
                        ${modeBadgeText}
                    </div>
                </div>
                <div class="card-content">
                    <div class="card-badges">
                        <span class="timestamp-badge">⏱ ${startFmt} - ${endFmt} (${clip.duration}s)</span>
                        <span class="score-badge">Clip #${clip.clip_number}</span>
                    </div>
                    <p class="transcript-quote" title="${escapeHtml(clip.text)}">“${escapeHtml(clip.text)}”</p>
                    <div class="card-actions">
                        <a href="${clip.video_url}" download="clipforge_clip_${clip.clip_number}.mp4" class="btn-download-video">
                            Download Video
                        </a>
                        <a href="${clip.srt_url}" download="clipforge_subtitles_${clip.clip_number}.srt" class="btn-download-srt">
                            SRT
                        </a>
                    </div>
                </div>
            `;
            clipsContainer.appendChild(card);
        });

        // Setup 1-Click Copy Title buttons
        clipsContainer.querySelectorAll(".btn-copy-title").forEach(btn => {
            btn.addEventListener("click", async () => {
                const clipId = btn.getAttribute("data-clip-id");
                const input = document.getElementById(`input-title-${clipId}`);
                if (!input) return;

                const textToCopy = input.value.trim();
                if (!textToCopy) return;

                try {
                    if (navigator.clipboard && window.isSecureContext) {
                        await navigator.clipboard.writeText(textToCopy);
                    } else {
                        // Fallback selection copy for non-https/local dev
                        input.select();
                        document.execCommand("copy");
                    }

                    const copyText = btn.querySelector(".copy-text");
                    const copyIcon = btn.querySelector(".copy-icon");
                    const originalText = copyText ? copyText.textContent : "Copy Title";
                    const originalIcon = copyIcon ? copyIcon.textContent : "📋";

                    btn.classList.add("copied");
                    if (copyText) copyText.textContent = "Copied!";
                    if (copyIcon) copyIcon.textContent = "✓";

                    setTimeout(() => {
                        btn.classList.remove("copied");
                        if (copyText) copyText.textContent = originalText;
                        if (copyIcon) copyIcon.textContent = originalIcon;
                    }, 2000);
                } catch (err) {
                    console.error("Failed to copy title:", err);
                    input.select();
                }
            });
        });

        // Persist title edits when user modifies input
        clipsContainer.querySelectorAll(".clip-title-input").forEach(input => {
            input.addEventListener("change", async () => {
                const clipId = input.getAttribute("data-clip-id");
                const newTitle = input.value.trim();
                if (!clipId || !newTitle || !currentJobId) return;

                try {
                    await fetch(`/api/jobs/${currentJobId}/clips/${clipId}/title`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ title: newTitle })
                    });
                } catch (err) {
                    console.warn("Could not persist title edit to server:", err);
                }
            });
        });

        // Smooth scroll down to results
        clipsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function setValidating(isValidating) {
        btnValidateUrl.disabled = isValidating;
        if (isValidating) {
            validateText.classList.add("hidden");
            validateSpinner.classList.remove("hidden");
        } else {
            validateText.classList.remove("hidden");
            validateSpinner.classList.add("hidden");
        }
    }

    function setLoading(isLoading) {
        generateButton.disabled = isLoading;
        if (isLoading) {
            btnText.textContent = "Processing YouTube Video...";
            btnSpinner.classList.remove("hidden");
        } else {
            btnText.textContent = "Generate Vertical Shorts";
            btnSpinner.classList.add("hidden");
        }
    }

    function showStatus(message, isError = false) {
        statusMessage.textContent = message;
        statusMessage.className = `status-msg ${isError ? 'error' : 'success'}`;
    }

    function formatSeconds(secs) {
        const m = Math.floor(secs / 60);
        const s = Math.floor(secs % 60);
        return `${m}:${s.toString().padStart(2, "0")}`;
    }

    function escapeHtml(text) {
        if (!text) return "";
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});