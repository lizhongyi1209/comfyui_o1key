import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "o1key.renameTab",
    async setup() {
        const rename = () => {
            if (document.title.includes("ComfyUI")) {
                document.title = document.title.replace("ComfyUI", "o1key");
            }
        };

        rename();
        new MutationObserver(rename).observe(
            document.querySelector("title") || document.head,
            { childList: true, subtree: true, characterData: true }
        );
    },
});
