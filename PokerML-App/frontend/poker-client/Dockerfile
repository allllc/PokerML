# Frontend Dockerfile - Multi-stage build for production
# NOTE: Build from project root with: docker build -f frontend/poker-client/Dockerfile .

# Stage 1: Build the React application
# Using node:20 instead of node:20-alpine to avoid permission issues
FROM node:20 AS builder

WORKDIR /app

# Copy package files
COPY frontend/poker-client/package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY frontend/poker-client/ .

# Build argument for API URL (set at build time)
ARG VITE_API_BASE
ENV VITE_API_BASE=${VITE_API_BASE}

# Fix permissions on node_modules binaries and build the application
RUN chmod -R +x node_modules/.bin/ && \
    ./node_modules/.bin/tsc -b && \
    ./node_modules/.bin/vite build

# Stage 2: Serve with nginx
FROM nginx:alpine

# Replace the entire nginx config (not just conf.d)
COPY frontend/poker-client/nginx.conf /etc/nginx/nginx.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
