# AGENTS.md

## Project overview
This repository contains a Streamlit-based research assistant that uses LangChain agents with OpenRouter models and external tools (Tavily, Wikipedia, arXiv). The app is implemented in `app.py` and is designed to answer user questions by using tools when needed and by optionally grounding responses in uploaded PDF content.

## How to run
- Install dependencies:
  - `python -m pip install -r requirements.txt.txt`
- Start the app:
  - `streamlit run app.py`

## Key architecture and conventions
- `app.py` is the full application entry point.
- The app uses a Streamlit sidebar for runtime configuration:
  - OpenRouter API key
  - Tavily API key (optional)
  - model selection
  - PDF upload
- Tool choices are created inside the chat flow, not at module import time.
- The agent is built with `create_tool_calling_agent(...)` and `AgentExecutor(...)` from LangChain.
- PDF text is extracted with `pypdf` and injected into the system prompt as context when present.
- Chat state is stored in `st.session_state.messages` and `st.session_state.chat_history`.

## Project-specific guidance for AI agents
- Prefer preserving the current Streamlit UX and sidebar configuration pattern when editing the app.
- When changing the agent behavior, keep the system prompt logic and tool routing aligned with the existing research-agent design.
- Do not remove the requirement to cite tool-based information with source URLs or reference identifiers in the final answer.
- If Tavily is unavailable, the app should still function with Wikipedia and arXiv tools.
- Keep the PDF injection flow intact: uploaded PDF content should be added to the system prompt only after extraction and trimming.
- The dependency file is named `requirements.txt.txt`; use that exact path for install commands unless there is a deliberate cleanup change.

## Common pitfalls
- The app requires a valid OpenRouter API key before any chat request can proceed.
- If the user uploads a PDF, the extracted text is truncated to roughly 15,000 characters to avoid token overflow.
- `st.rerun()` is used after clearing the conversation, so UI resets remain consistent with the app’s state model.
- Tool calls are capped with `max_iterations=5` to avoid runaway loops.

## Typical tasks
- Update model/provider configuration: modify the model list in the sidebar and ensure the selected model is compatible with the OpenRouter base URL.
- Add or remove tools: edit the `tools` list and keep the system prompt description in sync.
- Change answer formatting: update the final output rendering in the chat response block without breaking the stored chat history.
- Improve PDF context handling: maintain the extraction helper and the prompt injection pattern.

## References
- App entry: `app.py`
- Dependencies: `requirements.txt.txt`
