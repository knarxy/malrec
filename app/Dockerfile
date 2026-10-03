# --- build ---------------------------------------------------------------
FROM docker.io/library/node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
COPY public ./public
RUN npm run build

# --- serve ---------------------------------------------------------------
FROM docker.io/library/nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY security-headers.conf /etc/nginx/security-headers.conf
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
