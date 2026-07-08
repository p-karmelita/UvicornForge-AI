# UnicornForge AI – Prompt Design (v0.1)

## 1. Goal of the Prompt

The prompt should guide the AI model to transform a rough project idea and basic context into a **complete, structured startup brief**.

Key requirements:

- Always return the **same 10 sections**.
- Use **clear, concise English**.
- Stay focused on **practical, startup-oriented content** suitable for hackathons and early-stage founders.

## 2. Input Variables

The backend will fill these placeholders:

- `{{project_idea}}` – main description (required).
- `{{target_users}}` – who this is for (optional).
- `{{industry}}` – domain or sector (optional).
- `{{available_time}}` – time constraints (optional).
- `{{available_technologies}}` – key tools/platforms (optional).

If any optional field is empty, the backend may pass a short placeholder like `"not specified"`.

## 3. Prompt Template (Version 0.1)
