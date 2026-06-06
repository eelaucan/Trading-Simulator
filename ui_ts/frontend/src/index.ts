import { DEMO_TRADE_PLANNER_PROPS } from "./demoProps.js";
import { TradePlannerApp } from "./TradePlanner.js";
import type { TradePlannerEventPayload, TradePlannerProps } from "./types.js";

const root = document.getElementById("root");
if (!root) {
  throw new Error("Trade planner root element was not found.");
}

const isStreamlitEmbed = (): boolean => {
  try {
    return window.parent !== window;
  } catch {
    return true;
  }
};

const postToStreamlit = (
  type: string,
  payload: Record<string, unknown>,
): void => {
  if (!isStreamlitEmbed()) {
    return;
  }

  window.parent.postMessage(
    {
      isStreamlitMessage: true,
      type,
      ...payload,
    },
    "*",
  );
};

const app = new TradePlannerApp(root, {
  emit: (payload: TradePlannerEventPayload): void => {
    if (isStreamlitEmbed()) {
      postToStreamlit("streamlit:setComponentValue", {
        value: payload,
        dataType: "json",
      });
      return;
    }

    console.info("Demo mode plan update", payload);
  },
  setFrameHeight: (): void => {
    window.requestAnimationFrame(() => {
      if (!isStreamlitEmbed()) {
        return;
      }

      const height = Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight,
      );
      postToStreamlit("streamlit:setFrameHeight", { height });
    });
  },
});

const onRender = (event: MessageEvent): void => {
  const payload = event.data;
  if (!payload || payload.type !== "streamlit:render") {
    return;
  }
  const nextProps: TradePlannerProps = {
    ...(payload.args as TradePlannerProps),
    disabled: Boolean(payload.disabled),
  };
  app.setProps(nextProps);
};

window.addEventListener("message", onRender);
window.addEventListener("load", () => {
  if (isStreamlitEmbed()) {
    postToStreamlit("streamlit:componentReady", { apiVersion: 1 });
    return;
  }

  document.body.classList.add("tp-standalone");
  app.setProps(DEMO_TRADE_PLANNER_PROPS);
});
