# API Key Security Setup Guide

## ⚠️ IMPORTANT SECURITY RULES

1. **NEVER** commit API keys to GitHub
2. **NEVER** hardcode API keys in source code
3. **ALWAYS** use environment variables or secrets management
4. **ALWAYS** add `.env` to `.gitignore`

---

## Option 1: Local Development (Python Scripts)

### Step 1: Create `.env` File

```bash
# In your project root directory
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025
cp .env.example .env
```

### Step 2: Add Your API Key to `.env`

Edit `.env` file and add your key:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

### Step 3: Load `.env` in Your Python Scripts

Add this to the top of your Python files:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Now you can access the API key
api_key = os.getenv('OPENAI_API_KEY')
```

### Step 4: Install `python-dotenv`

```bash
pip install python-dotenv
```

---

## Option 2: Google Colab (Recommended for Notebooks)

### Method A: Using Colab Secrets (Best Practice)

1. **Open your Colab notebook**
2. **Click the 🔑 key icon** in the left sidebar (Secrets)
3. **Click "Add new secret"**
4. **Set:**
   - Name: `OPENAI_API_KEY`
   - Value: `your-openai-api-key-here`
5. **Enable notebook access** (toggle the switch)
6. **Your code will automatically load it:**

```python
from google.colab import userdata
api_key = userdata.get('OPENAI_API_KEY')
```

### Method B: Using Environment Variables (Less Secure)

Add this cell at the top of your notebook (but DON'T save it):

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'
```

⚠️ **Delete this cell before saving the notebook!**

---

## Current Implementation

Your Colab notebooks already have the correct security pattern:

```python
# Step 1: Try environment variable (for local scripts)
api_key = os.getenv('OPENAI_API_KEY')

# Step 2: If not found, try Colab secrets (for notebooks)
if not api_key:
    try:
        from google.colab import userdata
        api_key = userdata.get('OPENAI_API_KEY')
    except:
        print("ERROR: OPENAI_API_KEY not found!")
        raise ValueError("OpenAI API key required")

# Step 3: Use the API key
pipeline = VannaPipeline(api_key=api_key)
```

This pattern works for **both** local development and Google Colab!

---

## What's Already Protected

✅ `.env` is in `.gitignore` (won't be committed)
✅ `.env.example` is provided as a template (safe to commit)
✅ Your code uses `os.getenv()` and Colab secrets (secure)

---

## Action Items

### For Local Development:
1. Copy `.env.example` to `.env`
2. Add your API key to `.env`
3. Install `python-dotenv`: `pip install python-dotenv`
4. Never commit `.env` to GitHub

### For Google Colab:
1. Open your Colab notebook
2. Click the 🔑 Secrets icon
3. Add `OPENAI_API_KEY` secret
4. Enable notebook access
5. Your code will automatically use it

---

## ⚠️ SECURITY WARNING

**Your API key was exposed in the conversation!** After setting this up:

1. Go to https://platform.openai.com/api-keys
2. **Revoke** the exposed key
3. **Generate a new key**
4. Add the new key to `.env` or Colab secrets
5. Never share your API key in chat, code, or screenshots

---

## Verification

Test that your API key is loaded correctly:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✅ API key loaded: {api_key[:20]}...{api_key[-4:]}")
else:
    print("❌ API key not found!")
```

This will show: `✅ API key loaded: your-openai-...here`
