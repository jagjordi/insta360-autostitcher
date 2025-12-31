FROM node:20-bullseye AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=${VITE_API_BASE}

RUN npm run build

FROM nginx:alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx-default.conf.template /etc/nginx/templates/default.conf.template

ENV NGINX_PORT=3000
ENV API_PROXY_TARGET=http://127.0.0.1:8000

EXPOSE 80
