# Market Research System v1.0 - Docker Container
# Multi-stage build for optimized production image
# Base: Python 3.9 on Ubuntu 20.04 LTS

# Build stage
FROM python:3.9-slim-bullseye as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0
ARG VCS_REF

# Add metadata labels
LABEL maintainer="market-research-team@company.com" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0" \
      org.label-schema.name="Market Research System" \
      org.label-schema.description="Automated market data analysis and reporting system" \
      org.label-schema.vendor="Independent Market Research"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim-bullseye as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sqlite3 \
    cron \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r marketresearch && \
    useradd -r -g marketresearch -d /home/marketresearch -s /bin/bash marketresearch && \
    mkdir -p /home/marketresearch && \
    chown -R marketresearch:marketresearch /home/marketresearch

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=marketresearch:marketresearch . .

# Create necessary directories
RUN mkdir -p /app/data/{raw,processed,cache,backups} \
             /app/reports/{daily,weekly,monthly} \
             /app/logs/{application,system,archived} \
             /var/log/market-research \
             /etc/market-research && \
    chown -R marketresearch:marketresearch /app \
                                          /var/log/market-research \
                                          /etc/market-research

# Copy configuration files
COPY --chown=marketresearch:marketresearch config/ /etc/market-research/
COPY tools/deployment/docker/supervisord.conf /etc/supervisor/conf.d/
COPY tools/deployment/docker/nginx.conf /etc/nginx/sites-available/market-research
COPY tools/deployment/docker/crontab /etc/cron.d/market-research

# Configure cron
RUN chmod 0644 /etc/cron.d/market-research && \
    crontab -u marketresearch /etc/cron.d/market-research

# Configure nginx
RUN ln -sf /etc/nginx/sites-available/market-research /etc/nginx/sites-enabled/ && \
    rm -f /etc/nginx/sites-enabled/default

# Set permissions
RUN chmod +x /app/scripts/*.py /app/tools/deployment/docker/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8081

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MARKET_RESEARCH_ENV=production \
    MARKET_RESEARCH_CONFIG=/etc/market-research

# Switch to non-root user
USER marketresearch

# Set entrypoint
ENTRYPOINT ["/app/tools/deployment/docker/entrypoint.sh"]

# Default command
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]