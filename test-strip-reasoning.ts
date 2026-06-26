import { convertToModelMessages, type UIMessage } from "ai";
const msgs = [{ role: "assistant", content: "foo", parts: [{ type: "text", text: "foo" }, { type: "reasoning", text: "hmm" }] }];
const core = convertToModelMessages(msgs as UIMessage[]);
console.log(JSON.stringify(core, null, 2));
