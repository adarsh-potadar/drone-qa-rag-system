# 🎯 Complete Setup Guide (2-3 Hours)

## Phase 1: Environment Setup (30 minutes)

### Step 1: Install Ollama
```bash
# For macOS/Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# For Windows:
# Download from: https://ollama.ai/download/windows
```

### Step 2: Download Llama2 Model
```bash
ollama pull llama2
# This downloads ~4GB model, may take 10-20 mins
```

### Step 3: Test Ollama
```bash
ollama run llama2
# Type: "Hello" and press Enter
# If it responds, you're good! Type /bye to exit
```

### Step 4: Create Project Structure
```bash
# Create project folder
mkdir drone-qa-rag
cd drone-qa-rag

# Create subfolders
mkdir data
mkdir screenshots

# Create virtual environment
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 5: Install Dependencies
```bash
# Copy requirements.txt to your project folder first
pip install -r requirements.txt

# This may take 5-10 minutes
# Downloads ~2GB of models and libraries
```

---

## Phase 2: Get Drone Manual (15 minutes)

### Option 1: DJI Manual (Recommended - Easy)
1. Go to: https://www.dji.com/support
2. Click any drone (e.g., "DJI Mini 3")
3. Scroll to "Downloads" → Click "User Manual"
4. Download PDF
5. Rename to `drone_manual.pdf`
6. Move to `data/` folder

### Option 2: ArduPilot Documentation
1. Go to: https://ardupilot.org/copter/
2. Click "Download PDF" (usually in top right)
3. Save as `drone_manual.pdf` in `data/` folder

### Option 3: Quick Test PDF
If you can't find drone manual, use ANY technical PDF for testing:
- Any electronics manual
- Raspberry Pi documentation
- Arduino guides

**Just make sure it's named `drone_manual.pdf` and in `data/` folder!**

---

## Phase 3: Create Code Files (30 minutes)

Copy these files from the artifacts I created:

1. **rag_pipeline.py** - Core RAG logic
2. **app.py** - Streamlit UI
3. **requirements.txt** - Dependencies
4. **README.md** - Documentation

Your folder should look like:
```
drone-qa-rag/
├── venv/
├── data/
│   └── drone_manual.pdf
├── app.py
├── rag_pipeline.py
├── requirements.txt
└── README.md
```

---

## Phase 4: Test & Run (45 minutes)

### Step 1: Test RAG Pipeline (Command Line)
```bash
# Make sure Ollama is running in another terminal:
ollama run llama2

# Run the pipeline test
python rag_pipeline.py

# You should see:
# - Loading PDF...
# - Creating embeddings...
# - Testing questions...
# - Answers!
```

### Step 2: Run Streamlit App
```bash
streamlit run app.py
```

Browser will open automatically at `http://localhost:8501`

### Step 3: Build Pipeline in UI
1. Click "🔨 Build Pipeline" in sidebar
2. Wait 1-2 minutes
3. Green success message appears!

### Step 4: Test Questions
Try these questions:
- "How do I calibrate the compass?"
- "What is the maximum flight altitude?"
- "How do I start the drone?"

### Step 5: Take Screenshots
- Take 3-4 screenshots of:
  - The UI with a question
  - The answer with sources
  - Different questions
- Save in `screenshots/` folder

---

## Phase 5: GitHub Upload (30 minutes)

### Step 1: Create .gitignore
```bash
# Create .gitignore file
cat > .gitignore << EOF
venv/
__pycache__/
*.pyc
vectorstore/
.env
*.pdf
EOF
```

### Step 2: Initialize Git
```bash
git init
git add .
git commit -m "Initial commit: Drone Q&A RAG system"
```

### Step 3: Create GitHub Repo
1. Go to github.com
2. Click "New Repository"
3. Name: `drone-qa-rag`
4. Description: "RAG-powered chatbot for drone manuals using LangChain, HuggingFace, FAISS, and Ollama"
5. Keep it Public
6. Don't initialize with README
7. Click "Create"

### Step 4: Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/drone-qa-rag.git
git branch -M main
git push -u origin main
```

---

## Phase 6: Resume & Demo Prep (30 minutes)

### Update Resume - Projects Section:

```
DRONE MANUAL Q&A ASSISTANT | RAG System
- Built end-to-end RAG pipeline using LangChain for semantic search over drone documentation
- Implemented document chunking, embedding (HuggingFace), and vector storage (FAISS)
- Integrated Ollama (Llama2) for local LLM inference with zero API costs
- Created interactive Streamlit UI with source citation for transparency
- Tech: Python, LangChain, HuggingFace, FAISS, Ollama, Streamlit
- GitHub: github.com/YOUR_USERNAME/drone-qa-rag
```

### Prepare Demo Script (2 minutes):

**"Let me show you my RAG project:**
1. I built this Q&A system for drone manuals
2. It uses LangChain to load PDFs, chunk them, and embed them using HuggingFace
3. FAISS stores the vectors for fast retrieval
4. When you ask a question, it finds relevant chunks and uses Ollama's Llama2 to generate answers
5. See here - it shows the exact source pages used
6. Everything runs locally, no API costs
7. Built this in 2-3 hours to understand RAG end-to-end"

---

## ⚠️ Common Issues & Fixes

### Ollama not connecting?
```bash
# Check if Ollama is running:
ollama list

# Restart it:
ollama run llama2
```

### Module not found errors?
```bash
# Make sure virtual environment is activated:
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies:
pip install -r requirements.txt
```

### PDF not loading?
- Check PDF is in `data/` folder
- Check filename is exactly `drone_manual.pdf`
- Try a different PDF

### Out of memory?
- Close other applications
- Or edit `rag_pipeline.py` and reduce `chunk_size` to 500

---

## ✅ Success Checklist

Before applying to Flytbase:
- [ ] Project runs successfully
- [ ] Can answer at least 3 questions correctly
- [ ] Screenshots taken
- [ ] Code pushed to GitHub
- [ ] README is complete
- [ ] Can explain the RAG pipeline in 2 minutes
- [ ] Resume updated with project

---

## 🎯 Interview Prep

**Be ready to answer:**

1. **"Walk me through your RAG project"**
   - Mention: document loading, chunking, embeddings, vector store, retrieval, generation
   
2. **"Why did you choose these technologies?"**
   - Free/local (Ollama), popular in industry (LangChain), fast search (FAISS)
   
3. **"What challenges did you face?"**
   - First time: setting up Ollama, understanding chunk sizes, handling PDF parsing
   
4. **"How would you improve it?"**
   - Add multiple PDFs, better chunk strategy, evaluation metrics, deploy to cloud

---

**You got this! 🚀 Start with Phase 1 and work through each step. Any issues, just ask me!**