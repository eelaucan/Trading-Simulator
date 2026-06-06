async function request(path, init) {
    const response = await fetch(path, {
        headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
        ...init,
    });
    if (!response.ok) {
        const detail = await response.text();
        let message = detail || `Request failed (${response.status})`;
        try {
            const parsed = JSON.parse(detail);
            if (typeof parsed.detail === "string") {
                message = parsed.detail;
            }
        }
        catch {
            // Keep raw response text when the API does not return JSON.
        }
        throw new Error(message);
    }
    return (await response.json());
}
export async function fetchDatasets() {
    const payload = await request("/api/datasets");
    return payload.datasets;
}
export async function startSession(input) {
    return request("/api/session/start", {
        method: "POST",
        body: JSON.stringify(input),
    });
}
export async function advanceGeminiStep(session) {
    return request("/api/session/ai-step", {
        method: "POST",
        body: JSON.stringify({ session }),
    });
}
export async function sendPlannerEvent(session, event) {
    return request("/api/session/planner", {
        method: "POST",
        body: JSON.stringify({ session, event }),
    });
}
