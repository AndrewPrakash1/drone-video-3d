# Quick Fix: Deploy Backend to Connect Your Frontend

## The Problem

Your frontend is deployed at Vercel but can't reach the backend because the backend is not deployed yet. The error "Dashboard cannot reach the API at http://127.0.0.1:8765" means it's trying to connect to localhost, which doesn't work for a cloud deployment.

## The Solution: Deploy Backend to Render (5 Minutes)

### Step 1: Sign Up for Render

1. Go to **[render.com](https://render.com)**
2. Click **"Get Started"**
3. Sign up with your GitHub account (recommended for easy deployment)

### Step 2: Create a New Web Service

1. Click **"New +"** button in the dashboard
2. Select **"Web Service"**
3. Click **"Connect" next to your `drone-video-3d` repository**
   - If you don't see it, click "Configure account" to grant Render access

### Step 3: Configure the Service

Fill in these settings:

- **Name**: `onepass-backend` (or any name you prefer)
- **Region**: Choose closest to you
- **Branch**: `cursor/deployment-setup-603c` (or `main` after merging)
- **Root Directory**: *Leave blank*
- **Runtime**: `Python 3`
- **Build Command**: 
  ```bash
  pip install -r backend/requirements.txt
  ```
- **Start Command**:
  ```bash
  cd /opt/render/project/src && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
  ```

### Step 4: Add Environment Variable

Scroll down to **"Environment Variables"** section:

- Click **"Add Environment Variable"**
- **Key**: `PORT`
- **Value**: `8765`

### Step 5: Create Web Service

1. Click **"Create Web Service"** button at the bottom
2. Wait 3-5 minutes for Render to build and deploy
3. You'll get a URL like: `https://onepass-backend.onrender.com`

### Step 6: Update Vercel Frontend

1. Go to **[vercel.com](https://vercel.com)** and log in
2. Find your OnePass project
3. Go to **Settings** → **Environment Variables**
4. Add a new variable:
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**: `https://onepass-backend.onrender.com` (your Render URL)
   - **Environment**: Select all (Production, Preview, Development)
5. Click **"Save"**

### Step 7: Redeploy Frontend

1. Go to **Deployments** tab in Vercel
2. Click the **"..."** menu on the latest deployment
3. Click **"Redeploy"**
4. Wait ~1 minute for the redeployment

## Test It!

1. Visit your Vercel URL: `https://temporary-swift-apogee-yrh18ts.vercel.app`
2. The error should be gone!
3. Click **"Run proxy mission"** to test
4. Use **"Measure"** to verify the 20m eave measurement

## Important Notes

### Free Tier Limitations

- **Render Free Tier**: Service spins down after 15 minutes of inactivity
- **First request after spin-down**: Takes 30-60 seconds to wake up
- **Solution**: Upgrade to $7/month for always-on service

### Alternative: Deploy Backend Elsewhere

If you prefer a different service, you can use:

- **Railway**: Auto-detects the Dockerfile, very simple
- **Fly.io**: Good for Docker deployments
- **Heroku**: Classic PaaS option
- **AWS/GCP/Azure**: More complex but powerful

The key is to get a public URL for your backend, then update `NEXT_PUBLIC_API_URL` in Vercel.

## Troubleshooting

### Backend Build Fails

If Render shows a build error:
1. Check the build logs
2. Common issues:
   - Python version mismatch (should be 3.11+)
   - Missing dependencies in requirements.txt
3. Push fixes to GitHub and Render will auto-redeploy

### Frontend Still Shows Error

1. Make sure you added the environment variable to Vercel
2. Make sure you redeployed the frontend
3. Check browser console for actual API URL it's trying to reach
4. Verify backend is running by visiting: `https://your-backend.onrender.com/health`

### Backend is Slow

This is normal on free tier:
- First request wakes up the service (30-60s)
- Subsequent requests are fast
- Upgrade to paid tier ($7/mo) for always-on service

## Need Help?

Check the full deployment guide in `DEPLOYMENT.md` for more options and detailed instructions.
