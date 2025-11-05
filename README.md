## Quick Start

### Project Files
- **main.py**: Entry point; initializes environment, builds vector store, wires model and launches UI.
- **document_utils.py**: Document pipeline utilities (download, load PDFs, split, build vector store).
- **agent_utils.py**: RAG agent/tool functions bound to a model and vector store.
- **gradio_app.py**: Gradio UI function returning the `ChatInterface`.
- **documents/**: Local PDFs and `urls.txt` for auto-downloads.
- **prompts.env**: Optional env file for changing `SYSTEM_PROMPT`.

### Setup (Python 3.10)
```bash
# From the project root
python3.10 -m venv .venv  # or: python3 -m venv .venv (if python3 == 3.10)
source .venv/bin/activate  # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

### Configure Environment
- Ensure `OPENAI_API_KEY` is set, e.g. in `.env`.
- Optional: set `SYSTEM_PROMPT`, `OPENAI_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.

### Run
```bash
python3 main.py
```

## Optional Data Science Take-Home Assignment: Building a Clinical Knowledge Assistant for Psychiatry

**The Goal**

Welcome! We're building a tool to help psychiatrists make faster, safer medication decisions. When a doctor is with a patient, they often need to quickly check prescribing guidelines, drug interactions, or dosing information. Sifting through dense medical documents during a visit is impossible.

Your task is to build a proof-of-concept for a conversational AI assistant that can answer these questions instantly. You will create a RAG system that uses a set of expert clinical documents as its knowledge base.

**The Core Task: Build a Q&A System for Anxiety Medications**

We want you to build a system that can ingest the clinical documents listed below and answer a psychiatrist's questions accurately, citing the source of its information.

**The Knowledge Base (Source Documents)**

To keep this task focused, please limit your solution to the following six documents. These are expert-level guidelines and FDA labels related to the treatment of Generalized Anxiety Disorder (GAD). You can save these as PDFs or HTML to work with them.

1. **Clinical Guideline:** Oregon Health Authority's Medication Treatment Algorithm for Adults with GAD.
   - https://www.oregon.gov/oha/HPA/DSI-Pharmacy/MHCAGDocs/Narrative-Medication-Treatment-Algorithm-for-Adults-with-GAD.pdf
2. **FDA Label (SSRI):** Escitalopram (Lexapro)
   - https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/021323s047lbl.pdf
3. **FDA Label (SSRI):** Sertraline (Zoloft)
   - https://www.accessdata.fda.gov/drugsatfda_docs/label/2014/019839s080s083,020990s039s041lbl.pdf
4. **FDA Label (SNRI):** Venlafaxine XR (Effexor XR)
   - https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/020699s107lbl.pdf
5. **FDA Label (SNRI):** Duloxetine (Cymbalta)
   - https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/021427s049lbl.pdf
6. **FDA Label (Benzodiazepine):** Alprazolam (Xanax)
   - https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/021434s016lbl.pdf

**Technical Guidance**

- **Models:** You can use any model. Please let us know what environment you used.
- **UI:** A user interface is not required. You can show your work in a Jupyter Notebook where you input questions and get answers. If you'd like to build a simple Streamlit or Gradio interface, that's a bonus.
- **Focus:** We are most interested in your process and your knowledge of building a RAG pipeline (document loading, chunking, embedding, retrieval, prompt engineering). A thoughtful approach is more important than achieving perfect scores on every question.

**Evaluation Questions**

Please test your final system using the following questions. We will use these to evaluate your work.

1. What are the first-line recommended medications for Generalized Anxiety Disorder (GAD)?
2. What is the recommended starting dose and maximum dose of Escitalopram for an adult with GAD?
3. A patient has chronic liver disease. Are there any specific warnings or contraindications regarding the use of Duloxetine?
4. What is the FDA's black box warning regarding the use of Venlafaxine and other antidepressants in young adults?
5. What are the risks of combining a benzodiazepine like Alprazolam with opioids?
6. A patient is taking Sertraline and also an NSAID like ibuprofen for pain. Is there any risk I should be aware of?
7. What is Serotonin Syndrome, and what are some examples of drug combinations that increase its risk?
8. My patient did not respond to two different SSRIs. What is an evidence-based second-line or adjunctive medication to consider?
9. What are the main differences in the mechanism of action between an SSRI like Sertraline and an SNRI like Venlafaxine?
10. According to the guidelines, for how long should a patient continue medication after their GAD symptoms have gone away?

**What to Submit**

This project should take about 8 hours. Please send us:

1. **Your Code:** A Jupyter Notebook or script showing your entire process. Please keep it clean and add comments to explain your design choices.
2. **A Short Summary:** A brief 1-page write-up (PDF or Markdown). Please explain your approach, the tools you chose, any challenges you faced, and how you might improve the system if you had more time.