# AI Powered Smart Email Classifier - Agile Documentation

## 1. Product Backlog

| Epic | User Story | Description | Priority | Story Points | Acceptance Criteria |
|---|---|---|---|---|---|
| Preprocessing | As a Data Scientist, I want to clean the email text dataset so that my models train on noise-free data. | Load Hugging Face dataset, remove HTML, stopwords, and lower-case text. | High | 5 | Cleaned output contains no special characters or HTML tags. |
| Classification | As an ML Engineer, I want to train baseline classifiers (Logistic Regression, Naive Bayes) so that we have an initial accuracy benchmark. | Train a category classifier utilizing TF-IDF vectorization. | High | 8 | Model achieves >80% accuracy on the test set. |
| Classification | As an ML Engineer, I want to fine-tune a DistilBERT model so that text classification accuracy improves handling complex context. | Use HuggingFace Trainer to fine-tune DistilBERT on categorized emails. | Medium | 13 | Training code executes without errors and improves baseline F1 score. |
| Urgency Detection | As a Product Owner, I want the system to detect urgency levels so that critical emails are handled firsthand. | Implement a rule-based heuristic and hybrid ML classifier. | High | 5 | "ASAP" or "Urgent" keywords correctly map to "High" confidence. |
| Dashboard | As an End User, I want an interactive dashboard so that I can visualize email predictions and stats without writing code. | Develop a Streamlit UI with input box, prediction display, category filters, and visualization charts. | Medium | 8 | UI displays predicted category, urgency, and charts successfully based on inferences. |
| API Development | As a Developer, I want a robust backend API so that the model can be queried programmatically by other enterprise nodes. | Create a `POST /predict` endpoint using FastAPI. | High | 5 | Endpoint returns `200 OK` with the correct JSON schema including `category` and `urgency`. |
| Deployment | As a DevOps Engineer, I want the project to be GitHub-ready and modularized so that it can be easily deployed to VMs or containers. | Write `README.md`, `requirements.txt`, and tests directory. | High | 3 | Project dependencies install flawlessly out-of-the-box locally. |

---

## 2. Sprint Planning

| Sprint | Sprint Goal | User Stories Selected | Tasks | Estimated Effort (Hrs) |
|---|---|---|---|---|
| **Sprint 1** | Establish Data Pipeline | Preprocessing | Load dataset, build regex cleaners, parse into train/test, save samples | 40 |
| **Sprint 2** | Prototype ML Models & NLP | Classification (Baseline & BERT) | Design TF-IDF pipeline, train LR/NB models, construct PyTorch Trainer loop | 50 |
| **Sprint 3** | Implement Urgency Engine & API | Urgency Detection, API Development | Write rule-based logic, build FastAPI backend, define Pydantic models, write Pytests | 40 |
| **Sprint 4** | Complete Full Stack & Docs | Dashboard, Deployment | Build Streamlit UI, create Matplotlib/Seaborn charts, finalize README.md and GitHub push | 40 |

---

## 3. Sprint Backlog

*(Example focused on sprinting components)*

| Sprint | Task Breakdown | Assigned Role | Status |
|---|---|---|---|
| Sprint 1 | Create `src/preprocessing.py` and logic | Data Scientist | Done |
| Sprint 1 | Write unit tests for data cleaner regex | Tester | Done |
| Sprint 2 | Train Baseline Logistic Regression & NB | Data Scientist | Done |
| Sprint 2 | DistilBERT fine-tuning pipeline script | Data Scientist | Done |
| Sprint 3 | Develop Hybrid Urgency logic mapping | Developer | Done |
| Sprint 3 | Implement `POST /predict` FastAPI | Developer | Done |
| Sprint 3 | API Unit tests (`test_api.py`) | Tester | Done |
| Sprint 4 | Streamlit layout and dashboard charts | Developer | Done |
| Sprint 4 | Finalize `README.md` & `requirements.txt` | DevOps Engineer | Done |

---

## 4. Daily Standup Sample

| Role | What was done yesterday? | What will be done today? | Blockers |
|---|---|---|---|
| Data Scientist | Finalized the tokenization pipeline for DistilBERT and mapped the target labels. | Training the model using HuggingFace Trainer on the 80/20 test split. | Currently observing GPU memory constraints; may need a smaller batch size. |
| Developer | Built the FastAPI structure and defined the `EmailRequest` schemas. | Connecting the trained `predict.py` module to the API endpoint and writing basic health checks. | Waiting on the Data Scientist to export the final `/models` `.pkl` objects. |
| Tester | Wrote basic regex validation tests in `test_api.py` | Writing End-to-End prediction tests using FastAPI `TestClient`. | None |

---

## 5. Burndown Chart Data

| Sprint Day | Ideal Remaining Tasks | Actual Remaining Tasks | Notes |
|---|---|---|---|
| Day 1 | 40 | 40 | Sprint kick-off |
| Day 2 | 36 | 38 | Initial environment setup took longer |
| Day 3 | 32 | 32 | Picked up pace, data loading logic done |
| Day 4 | 28 | 29 | Waiting on regex rules approval |
| Day 5 | 24 | 24 | Halfway check-in, on schedule |
| Day 6 | 20 | 18 | Finished module early |
| Day 7 | 16 | 16 | Unit testing in progress |
| Day 8 | 12 | 10 | Testing done smoothly |
| Day 9 | 8 | 5 | Clean code refactoring completed early |
| Day 10 | 0 | 0 | Sprint Demo & Retrospective |

---

## 6. Defect Tracker

| Defect ID | Description | Severity | Status | Assigned To |
|---|---|---|---|---|
| BUG-001 | Model accuracy low for 'Feedback' category | Medium | Closed | Data Scientist |
| BUG-002 | `ModuleNotFoundError` during `python -m src.train_model` due to missing `sys.path` | High | Closed | Developer |
| BUG-003 | API crashing; missing `accelerate` package in `requirements.txt` | High | Closed | DevOps Engineer |
| BUG-004 | Dashboard charts not updating dynamically with API inferences | Medium | Closed | Developer |

---

## 7. Test Plan Summary

| Test Case | Input | Expected Output | Actual Output | Status |
|---|---|---|---|---|
| Validate text cleaner | `"<b>Hi!</b>"` | `"hi"` | `"hi"` | Pass |
| API Health Check | `GET /health` | `HTTP 200 { "status": "ok" }` | `HTTP 200 { "status": "ok" }` | Pass |
| Predict Endpoint Mock | `POST /predict {"email": "Help!"}` | `JSON {"category": "...", "urgency": "High"}` | `JSON {"category": "Request", "urgency": "High"}` | Pass |
| Predict Endpoint Null | `POST /predict {"email": ""}` | API Error or fallback | Model predicts fallback class | Pass |

---

## 8. Definition of Done (DoD)

| Check | Criteria | Description |
|---|---|---|
| ✅ | **Code Completed** | All logic scripts merged into the `src/` and `app/` folders successfully executing. |
| ✅ | **Tested** | `pytest` coverage passes without errors locally for endpoints and preprocessing logic. |
| ✅ | **Documented** | Detailed `README.md` written outlining run instructions and environment setup steps. |
| ✅ | **Deployed/Review** | Code has been pushed to the GitHub repository seamlessly executing end-to-end. |
