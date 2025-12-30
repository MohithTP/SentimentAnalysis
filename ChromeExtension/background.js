// background.js

// Listen for messages from content scripts or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyzeSentiment") {
        analyzeSentiment(request.comments)
            .then(data => sendResponse({ success: true, data: data }))
            .catch(error => sendResponse({ success: false, error: error.message }));
        return true; // Will respond asynchronously
    }
});

async function analyzeSentiment(comments) {
    const API_URL = "http://localhost:8080/invocations"; // Local Docker
    // const API_URL = "YOUR_SAGEMAKER_ENDPOINT_URL"; // Production

    // SageMaker expects: {"text": ["comment1", "comment2"]}
    const payload = {
        text: comments
    };

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            let errorMessage = response.statusText;
            try {
                const errorData = await response.json();
                if (errorData.error) {
                    errorMessage = errorData.error;
                }
            } catch (e) {
                // Ignore json parse error
            }
            throw new Error(`API Error: ${errorMessage}`);
        }

        const result = await response.json();
        // Result format: {"predictions": ["joy", "sadness", ...]}
        return result;
    } catch (error) {
        console.error("Sentiment Analysis Error:", error);
        throw error;
    }
}
