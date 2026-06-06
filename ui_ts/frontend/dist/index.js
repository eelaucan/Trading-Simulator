import { TradingSimulatorApp } from "./App.js";
const root = document.getElementById("root");
if (!root) {
    throw new Error("Application root element was not found.");
}
const app = new TradingSimulatorApp(root);
void app.init();
