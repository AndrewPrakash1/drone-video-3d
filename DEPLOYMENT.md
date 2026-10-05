# OnePass Deployment Guide

This guide covers deploying both the frontend and backend of OnePass.

## Frontend Deployment (Vercel)

### Option 1: Claim the Temporary Deployment

A temporary deployment has been created at:
**https://temporary-swift-apogee-yrh18ts.vercel.app**

This deployment expires in 60 minutes. To claim it permanently:

1. Visit: https://vercel.com/claim-deployment?code=2162da18-a4c6-48d7-90ae-ba0ab4736617
2. Sign up or log in to Vercel
3. The deployment will become permanent and you'll get a custom URL

### Option 2: Deploy from Scratch

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Navigate to the frontend directory and deploy:
   ```bash
   cd frontend
   vercel --prod
   ```

3. Follow the prompts to link to your Vercel account

## Backend Deployment

The backend is a Python FastAPI application that requires:
- Python 3.11+
- FFmpeg
- Optional: CUDA for GPU-accelerated processing

### Option 1: Deploy to Render (Recommended for simplicity)

1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Use these settings:
   - **Name**: `onepass-backend`
   - **Root Directory**: Leave blank
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd /opt/render/project/src && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11

5. Add environment variable:
   - Key: `PORT`
   - Value: `8765`

6. Click "Create Web Service"

Render will automatically build and deploy your backend. You'll get a URL like `https://onepass-backend.onrender.com`

### Option 2: Deploy with Docker

Use the included `Dockerfile`:

```bash
docker build -t onepass-backend .
docker run -p 8765:8765 onepass-backend
```

Deploy this container to:
- **Railway**: Import from GitHub, it will auto-detect the Dockerfile
- **Fly.io**: `fly launch` and follow prompts
- **Google Cloud Run**: `gcloud run deploy`
- **AWS ECS/Fargate**: Follow AWS container deployment guide

## Connecting Frontend to Backend

After deploying the backend, you need to configure the frontend to use the backend API:

1. In your Vercel project settings, add an environment variable:
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**: Your backend URL (e.g., `https://onepass-backend.onrender.com`)

2. Redeploy the frontend for the changes to take effect

## Testing the Deployment

1. Visit your frontend URL
2. Click "Run proxy mission" to test the demo scene
3. Use the "Measure" tool to verify the 20m eave measurement
4. For full functionality (video uploads), the backend must be running and connected

## Important Notes

### Performance Considerations

- **Free tiers** work for testing but may have cold start delays
- **GPU acceleration** (CUDA, COLMAP) is not available on standard free plans
- For production with GPU support, consider:
  - AWS EC2 with GPU instances
  - Google Cloud with GPU-enabled VMs
  - Dedicated servers with NVIDIA GPUs

### File Storage

The current setup uses local file storage. For production:

1. Add cloud storage (AWS S3, Google Cloud Storage, etc.)
2. Update `backend/app/storage.py` to use cloud storage
3. Set storage credentials as environment variables

### Security

- Add authentication for production deployments
- Configure CORS properly in `backend/app/main.py`
- Use environment variables for secrets
- Enable HTTPS (automatically provided by Vercel and Render)

## Monitoring

After deployment:

- **Vercel**: Dashboard shows deployment logs and analytics
- **Render**: Logs tab shows backend logs and resource usage
- Set up alerts for errors and downtime

## Cost Estimates

**Free Tier (Testing)**:
- Vercel: Free hobby plan (100GB bandwidth/month)
- Render: Free tier with 750 hours/month (sleeps after inactivity)

**Production (with reasonable traffic)**:
- Vercel: $20/month (Pro plan)
- Render: $7-25/month depending on resources
- Total: ~$27-45/month

**With GPU Support**:
- AWS EC2 g4dn.xlarge: ~$0.526/hour = ~$378/month (if running 24/7)
- Consider serverless GPU options for lower costs

## Support

For deployment issues:
- Check Vercel logs: `vercel logs <deployment-url>`
- Check Render logs: In the Render dashboard
- Review application logs for errors

## Next Steps

1. Claim your temporary Vercel deployment
2. Deploy the backend to Render or another platform
3. Connect them via environment variables
4. Test the full functionality
5. (Optional) Set up custom domain in Vercel settings
