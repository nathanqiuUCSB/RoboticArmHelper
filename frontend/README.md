# Vision-Guided Robotic Arm - Frontend

A clean, modern frontend for showcasing the vision-guided robotic arm hackathon project.

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

2. **Run the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

3. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Build for Production

```bash
npm run build
npm start
```

## Customization

### Adding a Video Demo

Edit `components/VideoDemo.tsx` and uncomment the iframe section, replacing `YOUR_VIDEO_ID` with your YouTube video ID, or add a local video file.

### Updating Team Members

Edit `components/AboutTeam.tsx` and update the `teamMembers` array with your actual team information.

### Changing Colors

Edit `tailwind.config.js` to modify the accent color and other theme settings.

## Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **React** - UI library
