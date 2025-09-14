# Railway Deployment Guide

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Git Repository**: Push your code to GitHub/GitLab
3. **Google Colab**: Set up the Colab notebook for model inference

## Deployment Steps

### 1. Prepare Repository

```bash
# Ensure all files are committed
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### 2. Deploy to Railway

1. **Connect Repository**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure Environment Variables**:
   ```
   SKIP_LOCAL_MODELS=true
   PORT=8000
   PYTHONPATH=/app
   ```

3. **Railway will automatically**:
   - Detect the `Dockerfile`
   - Build the container
   - Deploy the application
   - Provide a public URL

### 3. Set Up Google Colab

1. **Upload Notebook**:
   - Upload `code/ColabNotebook/MSE_Thesis_Modular_Pipelines.py` to Google Colab
   - Enable GPU runtime: Runtime → Change runtime type → GPU

2. **Run Colab Notebook**:
   - Execute all cells to start the API server
   - Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)

3. **Update Frontend Configuration**:
   - In Railway dashboard, add environment variable:
   ```
   COLAB_API_URL=https://your-ngrok-url.ngrok.io
   ```

### 4. Verify Deployment

1. **Health Check**: Visit `https://your-app.railway.app/health`
2. **API Documentation**: Visit `https://your-app.railway.app/api/docs`
3. **Frontend**: Visit `https://your-app.railway.app/`

## Local Testing

Test the Docker build locally before deploying:

```bash
# Build the image
docker build -t vietnamese-nl2sql .

# Run the container
docker run -p 8000:8000 -e SKIP_LOCAL_MODELS=true vietnamese-nl2sql

# Test with docker-compose
docker-compose up --build
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Railway App   │    │  Google Colab    │    │   SQLite DB     │
│                 │    │                  │    │                 │
│  ┌───────────┐  │    │  ┌─────────────┐ │    │  ┌───────────┐  │
│  │ Frontend  │  │    │  │ PhoBERT     │ │    │  │ Products  │  │
│  │ (Next.js) │  │    │  │ SQLCoder    │ │    │  │ Reviews   │  │
│  └───────────┘  │    │  │ Models      │ │    │  │ Brands    │  │
│  ┌───────────┐  │    │  └─────────────┘ │    │  └───────────┘  │
│  │ Backend   │◄─┼────┼─►│ FastAPI      │ │    │                 │
│  │ (FastAPI) │  │    │  │ ngrok        │ │    │                 │
│  └───────────┘  │    │  └─────────────┘ │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SKIP_LOCAL_MODELS` | Use Colab for model inference | `true` |
| `PORT` | Application port | `8000` |
| `PYTHONPATH` | Python path | `/app` |
| `COLAB_API_URL` | Google Colab ngrok URL | Required |

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check `Dockerfile` syntax
   - Verify all dependencies in `requirements.txt`

2. **Runtime Errors**:
   - Check Railway logs
   - Verify environment variables
   - Ensure Colab notebook is running

3. **Database Issues**:
   - Database is included in container
   - Check file permissions
   - Verify SQLite file exists

### Monitoring

- **Railway Dashboard**: Monitor deployments, logs, and metrics
- **Health Endpoint**: `/health` for application status
- **API Docs**: `/api/docs` for API testing

## Production Considerations

1. **Security**:
   - Configure CORS appropriately
   - Use environment variables for secrets
   - Enable HTTPS (Railway provides this automatically)

2. **Performance**:
   - Monitor response times
   - Consider caching strategies
   - Scale if needed using Railway's scaling options

3. **Reliability**:
   - Set up monitoring alerts
   - Configure automatic restarts
   - Keep Colab notebook running or implement fallbacks
