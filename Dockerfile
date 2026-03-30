FROM python:3.11

# Install Node.js 22 + nginx
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates nginx \
  && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for caching
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package-lock.json ./frontend/
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Install dependencies (Node + Python)
RUN npm ci \
  && npm ci --prefix frontend \
  && cd backend && uv sync

# Copy source code
COPY . .

# Build frontend as static files
RUN cd frontend && npm run build

# Nginx config: serve static + proxy /api to backend
RUN cat > /etc/nginx/sites-available/default <<'NGINX'
server {
    listen 3000;
    root /app/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
NGINX

EXPOSE 3000 5001

CMD ["sh", "-c", "nginx && cd backend && uv run python run.py"]
