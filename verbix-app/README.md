# Verbix Website

AI-Powered Indian Sign Language Translation Platform

## Setup

```bash
npm install
npm run dev
```

## Add Your Demo Video

1. Upload your demo video to YouTube
2. Open `src/App.jsx`
3. Find the Demo section (around line 170)
4. Uncomment the iframe and replace `VIDEO_ID` with your YouTube video ID

```jsx
<iframe 
  className="absolute inset-0 w-full h-full"
  src="https://www.youtube.com/embed/YOUR_VIDEO_ID"
  title="Verbix Demo"
  frameBorder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowFullScreen
/>
```

## Deploy to Vercel

```bash
npm install -g vercel
vercel
```

Or connect your GitHub repo to Vercel for automatic deployments.
