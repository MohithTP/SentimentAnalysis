// content.js
console.log("YouTube Sentiment Extension Loaded");

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getComments") {
        const comments = scrapeComments();
        sendResponse({ comments: comments });
    }
});

function scrapeComments() {
    const comments = [];
    // YouTube comments are in <ytd-comment-thread-renderer>
    // The text content is in #content-text
    const commentElements = document.querySelectorAll("ytd-comment-view-model"); // YouTube UI updates frequently, might need adjustment

    // Fallback selectors if the primary one fails (YouTube changes DOM often)
    const selectors = [
        "#content-text",
        ".yt-core-attributed-string"
    ];

    commentElements.forEach(el => {
        let text = "";
        for (let sel of selectors) {
            const textEl = el.querySelector(sel);
            if (textEl) {
                text = textEl.innerText;
                break;
            }
        }

        if (text) {
            // Clean up slightly
            text = text.replace(/\n/g, " ").trim();
            if (text.length > 0) comments.push(text);
        }
    });

    // Limit to top 50 for performance during testing
    return comments.slice(0, 50);
}
