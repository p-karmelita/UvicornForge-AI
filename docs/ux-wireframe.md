# UnicornForge AI – UX Wireframe (MVP)

This document describes the initial UX and layout for the MVP in text form, to align frontend and backend work.

## 1. Page Layout (Single Page)

The app is a **single-page interface** with two main panels:

- **Left (or top on mobile): Input Panel**
- **Right (or bottom on mobile): Output Panel**

### 1.1. Header

- **Title:** `UnicornForge AI`
- **Subtitle/tagline:** `Turn a rough idea into a startup brief in one click.`

### 1.2. Panels

- On larger screens (desktop/tablet):
  - Two-column layout:
    - Left: input form.
    - Right: generated brief.
- On smaller screens (mobile):
  - Stacked layout:
    - Input panel on top.
    - Output panel below.

## 2. Input Panel (Left / Top)

Elements (in order):

1. **Section title:** `Input`
2. **Field: Project idea (required)**
   - Label: `Project idea *`
   - Control: textarea (4–6 lines)
   - Placeholder example:
     > Describe your idea in a few sentences…

3. **Field: Target users (optional)**
   - Label: `Target users (optional)`
   - Control: single-line text input
   - Placeholder: `e.g. hackathon teams, students, founders`

4. **Field: Industry (optional)**
   - Label: `Industry (optional)`
   - Control: single-line text input
   - Placeholder: `e.g. education, healthcare, productivity`

5. **Field: Available time (optional)**
   - Label: `Available time (optional)`
   - Control: single-line text input
   - Placeholder: `e.g. 24 hours, one weekend`

6. **Field: Available technologies (optional)**
   - Label: `Available technologies (optional)`
   - Control: single-line text input
   - Placeholder: `e.g. AMD GPUs, Fireworks AI API, Python`

7. **Primary action button**
   - Label: `Generate Startup Brief`
   - Behavior:
     - Disabled while a generation request is in progress.
     - Triggers the call to the backend.

8. **Status text**
   - Small text element below the button.
   - Used for showing:
     - `Generating...`
     - `Done.`
     - `Error: please try again.`

## 3. Output Panel (Right / Bottom)

Elements:

1. **Section title:** `Generated Startup Brief`

2. **Secondary action button:** `Copy as Markdown`
   - Located near the top of the output panel.
   - Copies the entire generated brief in Markdown format to the clipboard.

3. **Content area**
   - Scrollable container with headings and text.
   - Default state (before generation):
     - Text: `(Waiting for input...)`
   - After a successful generation:
     - Show sections in this order:

       1. **Project name**
       2. **One-sentence pitch**
       3. **Problem**
       4. **Solution**
       5. **Target market**
       6. **MVP scope**
       7. **Key features**
       8. **Demo scenario**
       9. **Business model**
       10. **Why this project can win a hackathon**

   - Each section should have:
     - A clear heading (e.g., `## Problem`).
     - A short paragraph or list as content.

## 4. States and Feedback

### 4.1. Idle State

- Input form is empty or partially filled.
- Output panel shows `(Waiting for input...)`.
- Status text is empty.

### 4.2. Loading State

- Triggered after clicking `Generate Startup Brief`.
- Button is disabled.
- Status text: `Generating your startup brief…`.
- Output may still show the previous brief or the waiting text; this is acceptable for MVP.

### 4.3. Success State

- Button enabled again.
- Status text: `Done.` or a short success message.
- Output panel displays the new brief with all 10 sections.

### 4.4. Error State

- Button enabled again.
- Status text: `Error: please try again.` (or similar).
- Output panel may keep the last successful brief or show an error note.
- No modal dialogs are required for MVP; simple inline text is enough.

## 5. Visual Priorities

- The **Project idea** field and the **Generate** button should be visually prominent.
- The **Generated Startup Brief** section should:
  - occupy sufficient space,
  - make it clear that it is scrollable if long,
  - emphasize headings for readability.

## 6. Accessibility & Responsiveness (MVP Level)

- Labels should be explicitly connected to inputs.
- Color contrast should be sufficient for text.
- On small screens:
  - Panels stack vertically.
  - Buttons remain easy to tap.

## 7. Future Enhancements (Beyond MVP)

Not required for the first branch, but useful to keep in mind:

- Button to **regenerate** with the same input.
- Ability to edit sections inline.
- Tabs or modes for:
  - “Startup brief”
  - “Pitch deck outline”
  - “Demo script”
- Option to toggle between “short” and “detailed” brief.