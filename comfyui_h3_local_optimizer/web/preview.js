import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "h3.local.skill.prompt.preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["H3LocalSkillOptimizer", "H3LocalPromptPreview"].includes(nodeData.name)) return;
        const original = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            original?.apply(this, arguments);
            if (!this.h3ResultWidgets) {
                this.h3ResultWidgets = ["最终提示词", "引用映射"].map((name) => {
                    const widget = ComfyWidgets.STRING(this, name,
                        ["STRING", { multiline: true }], app).widget;
                    widget.inputEl.readOnly = true;
                    widget.inputEl.style.opacity = "0.9";
                    widget.options.serialize = false;
                    return widget;
                });
            }
            this.h3ResultWidgets[0].value = (message.text || []).join("\n");
            this.h3ResultWidgets[1].value = (message.mapping || []).join("\n");
            this.setSize([Math.max(this.size[0], 480), Math.max(this.size[1], 620)]);
            this.setDirtyCanvas(true, true);
        };
    },
});
