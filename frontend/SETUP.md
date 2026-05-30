# Frontend Setup Instructions
# Run these commands from the frontend/ directory after installing Node.js 20+

# 1. Install dependencies
npm install

# 2. Copy environment file
cp ../.env.example .env.local
# Edit .env.local: set NEXT_PUBLIC_API_URL=http://localhost:8000

# 3. Run dev server
npm run dev
# → Opens at http://localhost:3000

# 4. For production build
npm run build
npm start
