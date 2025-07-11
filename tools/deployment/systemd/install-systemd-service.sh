#!/bin/bash

# Market Research System v1.0 - Systemd Service Installation Script
# This script installs and configures systemd services for the market research system

set -e

# Configuration
SERVICE_USER="market-research"
SERVICE_GROUP="market-research"
INSTALL_DIR="/opt/market-research-v1"
SYSTEMD_DIR="/etc/systemd/system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Create service user and group
create_service_user() {
    log "Creating service user and group..."
    
    if ! getent group "$SERVICE_GROUP" > /dev/null; then
        groupadd --system "$SERVICE_GROUP"
        success "Created group: $SERVICE_GROUP"
    else
        warning "Group $SERVICE_GROUP already exists"
    fi
    
    if ! getent passwd "$SERVICE_USER" > /dev/null; then
        useradd --system --gid "$SERVICE_GROUP" \
                --home-dir "$INSTALL_DIR" \
                --shell /bin/false \
                --comment "Market Research System Service User" \
                "$SERVICE_USER"
        success "Created user: $SERVICE_USER"
    else
        warning "User $SERVICE_USER already exists"
    fi
}

# Set up directory permissions
setup_directories() {
    log "Setting up directory permissions..."
    
    # Create directories if they don't exist
    mkdir -p "$INSTALL_DIR"/{data,logs,reports,cache}
    mkdir -p "$INSTALL_DIR"/data/{raw,processed,cache,backups}
    mkdir -p "$INSTALL_DIR"/logs/{application,system,archived}
    mkdir -p "$INSTALL_DIR"/reports/{daily,weekly,monthly,archives}
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
    
    # Set permissions
    chmod 755 "$INSTALL_DIR"
    chmod 750 "$INSTALL_DIR"/{data,logs,reports,cache}
    chmod 700 "$INSTALL_DIR"/data/cache
    chmod 700 "$INSTALL_DIR"/logs
    
    success "Directory permissions configured"
}

# Install systemd service files
install_service_files() {
    log "Installing systemd service files..."
    
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Copy service files
    cp "$script_dir/market-research.service" "$SYSTEMD_DIR/"
    cp "$script_dir/market-research-timer.timer" "$SYSTEMD_DIR/"
    cp "$script_dir/market-research-data-collection.service" "$SYSTEMD_DIR/"
    cp "$script_dir/market-research-analysis.service" "$SYSTEMD_DIR/"
    cp "$script_dir/market-research-maintenance.service" "$SYSTEMD_DIR/"
    cp "$script_dir/market-research-maintenance.timer" "$SYSTEMD_DIR/"
    
    # Set correct permissions
    chmod 644 "$SYSTEMD_DIR"/market-research*.service
    chmod 644 "$SYSTEMD_DIR"/market-research*.timer
    
    success "Service files installed"
}

# Reload systemd and enable services
configure_systemd() {
    log "Configuring systemd services..."
    
    # Reload systemd daemon
    systemctl daemon-reload
    
    # Enable main service
    systemctl enable market-research.service
    
    # Enable timers
    systemctl enable market-research-timer.timer
    systemctl enable market-research-maintenance.timer
    
    success "Systemd services configured and enabled"
}

# Create configuration template
create_config_template() {
    log "Creating configuration template..."
    
    local config_dir="$INSTALL_DIR/config"
    mkdir -p "$config_dir/system"
    
    cat > "$config_dir/system/service.yaml" << 'EOF'
# Market Research System Service Configuration
service:
  name: "Market Research System v1.0"
  version: "1.0.0"
  environment: "production"
  
  # Service behavior
  daemon_mode: true
  max_workers: 4
  
  # Scheduling
  timezone: "Asia/Kolkata"
  market_hours:
    start: "09:15"
    end: "15:30"
  
  # Data collection intervals
  data_collection:
    pre_market: "06:00"
    market_hours: "*/5 * * * *"  # Every 5 minutes during market hours
    post_market: "16:00"
  
  # Report generation
  reports:
    daily: "18:00"
    weekly: "Sun 20:00"
    monthly: "1st 22:00"
  
  # Maintenance
  maintenance:
    cleanup: "02:00"
    backup: "01:00"
    
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  max_size: "100MB"
  backup_count: 5
  
notifications:
  email:
    enabled: true
    smtp_server: "localhost"
    smtp_port: 587
    
alerts:
  system_health: true
  data_issues: true
  analysis_errors: true
EOF
    
    chown "$SERVICE_USER:$SERVICE_GROUP" "$config_dir/system/service.yaml"
    chmod 640 "$config_dir/system/service.yaml"
    
    success "Configuration template created"
}

# Display service status
show_status() {
    log "Service installation completed!"
    echo
    echo "=== Service Status ==="
    systemctl status market-research.service --no-pager -l || true
    echo
    echo "=== Timer Status ==="
    systemctl list-timers market-research* --no-pager || true
    echo
    echo "=== Next Steps ==="
    echo "1. Configure API keys in $INSTALL_DIR/config/"
    echo "2. Test the installation:"
    echo "   sudo systemctl start market-research.service"
    echo "3. Check logs:"
    echo "   sudo journalctl -u market-research.service -f"
    echo "4. Enable automatic startup:"
    echo "   sudo systemctl start market-research-timer.timer"
    echo
}

# Main installation process
main() {
    log "Starting Market Research System service installation..."
    
    check_root
    create_service_user
    setup_directories
    install_service_files
    configure_systemd
    create_config_template
    show_status
    
    success "Installation completed successfully!"
}

# Run main function
main "$@"