async function request(path, init, attempt = 0) {
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
        const isTimeout = response.status === 504 ||
            /timed out|timeout|FUNCTION_INVOCATION_TIMEOUT/i.test(message);
        if (isTimeout && attempt < 2) {
            await new Promise((resolve) => setTimeout(resolve, 1200 * (attempt + 1)));
            return request(path, init, attempt + 1);
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
export async function advanceGeminiStep(session, maxSteps = 3) {
    return request("/api/session/ai-step", {
        method: "POST",
        body: JSON.stringify({ session, max_steps: maxSteps }),
    });
}
export async function sendPlannerEvent(session, event) {
    return request("/api/session/planner", {
        method: "POST",
        body: JSON.stringify({ session, event }),
    });
}
