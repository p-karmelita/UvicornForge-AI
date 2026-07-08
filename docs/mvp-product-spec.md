# UnicornForge AI – MVP Product Specification

## 1. Overview

UnicornForge AI is an AI-powered startup mentor that turns rough hackathon ideas into structured startup briefs. The MVP focuses on a single, clear flow: the user enters a short idea and context, clicks one button, and receives a complete startup brief ready to use in a pitch or demo.

## 2. Primary User

- **Persona:** Hackathon participant / early-stage founder / student.
- **Context:** Limited time (24–72 hours), small team, strong technical skills, less experience in product, market, and pitching.
- **Goal:** Quickly turn a vague idea into a clear, convincing startup concept.

## 3. Main MVP Use Case

> A user types in a rough description of a project, adds optional context (target users, industry, time, technologies), clicks “Generate Startup Brief”, and receives a structured brief that covers problem, solution, target market, MVP scope, key features, demo scenario, and business model.

## 4. User Flow (Step-by-Step)

1. **Landing on the page**
   - User sees a short explanation and the main input form.

2. **Entering the idea**
   - User fills in:
     - required field: project idea,
     - optional supporting context: target users, industry, available time, available technologies.

3. **Generating the brief**
   - User clicks the **“Generate Startup Brief”** button.
   - UI shows a loading state (e.g., “Generating…”).

4. **Viewing the result**
   - When the backend completes:
     - A structured startup brief is displayed with clearly separated sections in a fixed order.
     - User can scroll through all sections.

5. **Using the result**
   - User can:
     - read the brief,
     - copy it as Markdown (single click),
     - optionally tweak and regenerate (out of MVP scope for now).

## 5. Input Fields (MVP Contract)

Form fields and intended usage:

- **project_idea** (required)
  - Type: multiline text (textarea)
  - Description: short free-form description of the idea.
  - Example: “AI assistant that helps hackathon teams turn rough ideas into startup-ready project plans and pitches.”

- **target_users** (optional)
  - Type: single-line text
  - Description: who the solution is for.
  - Example: “hackathon teams, students, startup founders”

- **industry** (optional)
  - Type: single-line text
  - Description: domain or sector.
  - Example: “education technology, productivity tools”

- **available_time** (optional)
  - Type: single-line text
  - Description: time constraints for building the MVP.
  - Example: “24 hours”, “one weekend”

- **available_technologies** (optional)
  - Type: single-line text
  - Description: key technologies or platforms available.
  - Example: “AMD GPUs, Fireworks AI API, Python, React”

These fields form the **input contract** for both frontend and backend.

## 6. Output Sections (MVP Contract)

The backend should always return a structured brief with the following sections and keys. This is the **output contract** for the frontend.

1. **project_name**
   - Short, product-like name.
   - Example: “UnicornForge AI”

2. **one_sentence_pitch**
   - Single concise sentence explaining what the product does and for whom.
   - Example: “UnicornForge AI helps hackathon teams turn rough ideas into startup-ready briefs in minutes.”

3. **problem**
   - Description of the main user problem and context.

4. **solution**
   - High-level explanation of how the product addresses the problem.

5. **target_market**
   - Who will use it and where the opportunity lies.

6. **mvp_scope**
   - What realistically fits into the first version/MVP.

7. **key_features**
   - Bullet-style or paragraph listing of the most important features.

8. **demo_scenario**
   - A short, practical walkthrough that can be used in a live demo or video.

9. **business_model**
   - How this product could generate revenue or be sustained.

10. **why_it_can_win**
    - Why this project is a strong fit for the hackathon track and judges’ criteria.

The frontend should render these sections in the above order, with human-readable headings matching the key names.

## 7. Minimal UX Requirements

- **Single-page experience**
  - One main screen for both input and output. No navigation steps between pages.

- **Visibility of structure**
  - Output sections clearly labeled with headings (e.g., “Problem”, “Solution”, etc.).
  - Order of sections is consistent between runs.

- **Feedback**
  - When generating:
    - Visible loading text (e.g., “Generating your startup brief…”).
  - On error:
    - Short error message (“Something went wrong. Please try again.”).

- **Copy as Markdown**
  - One button that copies the entire brief as Markdown text.
  - The copied format should preserve headings and section separation for easy pasting into docs or pitch tools.

- **Mobile-friendly baseline**
  - Layout should stack vertically on small screens.
  - Core interaction (enter idea → generate → view brief) must remain usable on mobile.

## 8. Example Usage Scenarios

### Scenario A – Classic Hackathon Team

- **Input**
  - Project idea: “An AI assistant that generates startup briefs, MVP scopes, and demo scripts from rough hackathon ideas.”
  - Target users: “hackathon teams, students, early-stage founders”
  - Industry: “productivity tools for innovation”
  - Available time: “48 hours”
  - Available technologies: “AMD GPUs, Fireworks AI API”

- **Expected Output (high level)**
  - Project name suggesting speed and clarity.
  - Problem focused on teams struggling with product/market framing under time pressure.
  - Solution describing a guided, AI-driven startup brief.
  - MVP scope limited to a single-page generator.
  - Demo scenario describing a live use during a hackathon.

### Scenario B – University Student Project

- **Input**
  - Project idea: “Tool to help students turn thesis topics into startup concepts.”
  - Target users: “university students”
  - Industry: “edtech, startup education”
  - Available time: “one semester”
  - Available technologies: “Python, AMD GPU cluster, Fireworks AI API”

- **Expected Output (high level)**
  - Project name connected to learning or thesis.
  - Problem focused on students not knowing how to translate research into products.
  - MVP scope describing a simple UI plus AI generation.
  - Demo scenario tailored to a classroom/accelerator setting.

### Scenario C – Corporate Innovation Team

- **Input**
  - Project idea: “Internal AI tool to turn new initiative ideas into structured project briefs.”
  - Target users: “innovation managers, product leads”
  - Industry: “enterprise / corporate innovation”
  - Available time: “pilot within 4 weeks”
  - Available technologies: “AMD-powered cloud, internal tools, Fireworks AI”

- **Expected Output (high level)**
  - Problem emphasizing scattered, unstructured idea intake.
  - Solution focusing on structured briefs that feed into decision-making.
  - Business model and target market focusing on B2B / internal adoption.

These scenarios serve primarily as guidance for prompt design and output quality expectations.