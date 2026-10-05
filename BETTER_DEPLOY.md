# Better Deployment Options (No Spin-Down Issues)

You're right - Render's free tier spins down after 15 minutes of inactivity, which is annoying. Here are better alternatives:

---

## Option 1: Railway ⭐ RECOMMENDED

**Why Railway?**
- ✅ $5 free credit every month (enough for small backend 24/7)
- ✅ No spin-down issues
- ✅ Auto-deploys from GitHub
- ✅ Super simple setup

### Deploy to Railway (3 minutes)

1. **Sign up at [railway.app](https://railway.app)**
   - Use your GitHub account

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `drone-video-3d`
   - Select branch: `cursor/deployment-setup-603c`

3. **Railway auto-detects everything!**
   - It will find the Dockerfile or use the `railway.toml` config
   - It will automatically install dependencies
   - You'll get a URL like: `https://onepass-backend.up.railway.app`

4. **Add Environment Variables** (if needed)
   - Click on your service
   - Go to "Variables" tab
   - Add: `PORT=8765` (optional, Railway sets this automatically)

5. **Done!** Copy your Railway URL

### Connect to Vercel Frontend

1. Go to Vercel → Your project → Settings → Environment Variables
2. Add: `NEXT_PUBLIC_API_URL` = `https://your-backend.up.railway.app`
3. Redeploy frontend

**Cost**: FREE for small projects ($5/month credit), then ~$5-10/month if you exceed the credit

---

## Option 2: Fly.io (Also Good)

**Why Fly.io?**
- ✅ Generous free tier (3 VMs with 256MB RAM each)
- ✅ No spin-down on free tier
- ✅ Global deployment

### Deploy to Fly.io (5 minutes)

1. **Install Fly CLI**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**
   ```bash
   flyctl auth login
   ```

3. **Launch your app**
   ```bash
   cd /workspace
   flyctl launch
   ```
   - Answer the prompts:
     - App name: `onepass-backend` (or whatever you want)
     - Region: Choose closest to you
     - Deploy now: Yes

4. **Get your URL**
   ```bash
   flyctl status
   ```
   You'll get: `https://onepass-backend.fly.dev`

5. **Update Vercel**
   - Add `NEXT_PUBLIC_API_URL=https://onepass-backend.fly.dev` in Vercel
   - Redeploy

**Cost**: FREE for 3 small VMs

---

## Option 3: Vercel Backend (Same Platform as Frontend)

**Why Vercel for Backend?**
- ✅ Everything on one platform
- ✅ Serverless = no servers to manage
- ✅ Free tier is generous
- ❌ Requires converting FastAPI to serverless functions

This is more work but keeps everything unified. I can help set this up if you want.

---

## Option 4: Keep Render BUT Upgrade ($7/month)

If you like Render's interface:
- Upgrade to the $7/month plan
- Gets you always-on service (no spin-down)
- 512 MB RAM
- Simple and reliable

---

## My Recommendation

**For your use case: Go with Railway**

Why?
1. Free tier is actually usable ($5 credit/month)
2. No spin-down issues
3. Easiest deployment (auto-detects everything)
4. Can scale up easily when needed

**Steps:**
1. Sign up at railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Select your repo
4. Copy the URL
5. Add to Vercel as `NEXT_PUBLIC_API_URL`
6. Done!

---

## Cost Comparison

| Service | Free Tier | Spin-Down? | Cost if Exceeded |
|---------|-----------|------------|------------------|
| **Railway** | $5 credit/month | No | ~$5-10/month |
| **Fly.io** | 3 small VMs | No | $5-15/month |
| **Render** | 750 hours/month | Yes (15 min) | $7/month for always-on |
| **Vercel** | Generous | No (serverless) | $20/month (Pro) |

---

## Need Help?

Choose Railway and follow the 3-minute setup above. It's the simplest and most reliable free option.
