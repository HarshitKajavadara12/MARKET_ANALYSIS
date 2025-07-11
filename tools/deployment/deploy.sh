#!/bin/bash

# Market Research System v1.0 Deployment Script
# Automated deployment script for production environment
# Author: Market Research Team
# Date: 2022

set -e  # Exit on any error

# Configuration
APP_NAME="market-research"
APP_VERSION="1.0"
DEPLOY_USER="marketresearch"
DEPLOY_DIR="/opt/market-research"
SERVICE_NAME="market-research"
PYTHON_VERSION="3.9"
LOG_FILE="/var/log/market-research/deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    echo "[ERROR] $1" >> $LOG_FILE
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
    echo "[WARNING] $1" >> $LOG_FILE
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
    echo "[INFO] $1" >> $LOG_FILE
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
    fi
}

# Check system requirements
check_system() {
    log "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        error "This deployment script is designed for Linux systems"
    fi
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
    fi
    
    PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PYTHON_VER" < "3.8" ]]; then
        error "Python 3.8 or higher is required. Found: $PYTHON_VER"
    fi
    
    # Check required commands
    for cmd in pip3 git systemctl nginx; do
        if ! command -v $cmd &> /dev/null; then
            warning "$cmd is not installed. Some features may not work."
        fi
    done
    
    info "System requirements check completed"
}

# Create deployment user and directories
setup_user_and_dirs() {
    log "Setting up deployment user and directories..."
    
    # Create user if doesn't exist
    if ! id "$DEPLOY_USER" &>/dev/null; then
        sudo useradd -r -m -s /bin/bash $DEPLOY_USER
        info "Created user: $DEPLOY_USER"
    fi
    
    # Create directories
    sudo mkdir -p $DEPLOY_DIR
    sudo mkdir -p /var/log/market-research
    sudo mkdir -p /etc/market-research
    
    # Set ownership
    sudo chown -R $DEPLOY_USER:$DEPLOY_USER $DEPLOY_DIR
    sudo chown -R $DEPLOY_USER:$DEPLOY_USER /var/log/market-research
    sudo chown -R $DEPLOY_USER:$DEPLOY_USER /etc/market-research
    
    info "User and directories setup completed"
}

# Install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    # Create virtual environment
    if [[ ! -d "$DEPLOY_DIR/venv" ]]; then
        sudo -u $DEPLOY_USER python3 -m venv $DEPLOY_DIR/venv
        info "Created virtual environment"
    fi
    
    # Activate virtual environment and install packages
    sudo -u $DEPLOY_USER bash -c "
        source $DEPLOY_DIR/venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    "
    
    info "Dependencies installed successfully"
}

# Deploy application code
deploy_application() {
    log "Deploying application code..."
    
    # Create backup of existing deployment
    if [[ -d "$DEPLOY_DIR/app" ]]; then
        sudo -u $DEPLOY_USER cp -r $DEPLOY_DIR/app $DEPLOY_DIR/app.backup.$(date +%Y%m%d_%H%M%S)
        info "Created backup of existing deployment"
    fi
    
    # Copy application files
    sudo -u $DEPLOY_USER mkdir -p $DEPLOY_DIR/app
    sudo -u $DEPLOY_USER cp -r src/ $DEPLOY_DIR/app/
    sudo -u $DEPLOY_USER cp -r config/ $DEPLOY_DIR/app/
    sudo -u $DEPLOY_USER cp -r scripts/ $DEPLOY_DIR/app/
    sudo -u $DEPLOY_USER cp requirements.txt $DEPLOY_DIR/app/
    
    # Set permissions
    sudo chmod +x $DEPLOY_DIR/app/scripts/*.py
    
    # Create data directories
    sudo -u $DEPLOY_USER mkdir -p $DEPLOY_DIR/data/{raw,processed,cache,backups}
    sudo -u $DEPLOY_USER mkdir -p $DEPLOY_DIR/reports/{daily,weekly,monthly}
    sudo -u $DEPLOY_USER mkdir -p $DEPLOY_DIR/logs
    
    info "Application deployed successfully"
}

# Configure system services
configure_services() {
    log "Configuring system services..."
    
    # Copy systemd service files
    sudo cp tools/deployment/systemd/market-research.service /etc/systemd/system/
    sudo cp tools/deployment/systemd/market-research-timer.timer /etc/systemd/system/
    
    # Update service files with correct paths
    sudo sed -i "s|/opt/market-research|$DEPLOY_DIR|g" /etc/systemd/system/market-research.service
    sudo sed -i "s|marketresearch|$DEPLOY_USER|g" /etc/systemd/system/market-research.service
    
    # Reload systemd and enable services
    sudo systemctl daemon-reload
    sudo systemctl enable market-research.service
    sudo systemctl enable market-research-timer.timer
    
    info "System services configured"
}

# Configure log rotation
configure_logging() {
    log "Configuring log rotation..."
    
    # Create logrotate configuration
    sudo tee /etc/logrotate.d/market-research > /dev/null <<EOF
/var/log/market-research/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 $DEPLOY_USER $DEPLOY_USER
    postrotate
        systemctl reload market-research.service > /dev/null 2>&1 || true
    endscript
}
EOF
    
    info "Log rotation configured"
}

# Setup database
setup_database() {
    log "Setting up database..."
    
    # Run database initialization
    sudo -u $DEPLOY_USER bash -c "
        cd $DEPLOY_DIR/app
        source $DEPLOY_DIR/venv/bin/activate
        python -c 'from src.data.data_storage import DatabaseManager; DatabaseManager().initialize_database()'
    "
    
    info "Database setup completed"
}

# Configure firewall
configure_firewall() {
    log "Configuring firewall..."
    
    # Check if ufw is available
    if command -v ufw &> /dev/null; then
        # Allow SSH
        sudo ufw allow ssh
        
        # Allow web traffic if nginx is used
        if command -v nginx &> /dev/null; then
            sudo ufw allow 'Nginx Full'
        fi
        
        # Enable firewall
        sudo ufw --force enable
        info "Firewall configured"
    else
        warning "UFW not available, skipping firewall configuration"
    fi
}

# Run initial data collection
initial_data_collection() {
    log "Running initial data collection..."
    
    sudo -u $DEPLOY_USER bash -c "
        cd $DEPLOY_DIR/app
        source $DEPLOY_DIR/venv/bin/activate
        python scripts/data_collection/collect_historical_data.py --days 30
    "
    
    info "Initial data collection completed"
}

# Start services
start_services() {
    log "Starting services..."
    
    # Start main service
    sudo systemctl start market-research.service
    sudo systemctl start market-research-timer.timer
    
    # Check service status
    if sudo systemctl is-active --quiet market-research.service; then
        info "Market Research service started successfully"
    else
        error "Failed to start Market Research service"
    fi
    
    if sudo systemctl is-active --quiet market-research-timer.timer; then
        info "Market Research timer started successfully"
    else
        warning "Market Research timer may not have started properly"
    fi
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Check if service is running
    if sudo systemctl is-active --quiet market-research.service; then
        info "✓ Service is running"
    else
        error "✗ Service is not running"
    fi
    
    # Check if required directories exist
    for dir in "$DEPLOY_DIR/data" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/reports"; do
        if [[ -d "$dir" ]]; then
            info "✓ Directory exists: $dir"
        else
            error "✗ Directory missing: $dir"
        fi
    done
    
    # Check database connectivity
    if sudo -u $DEPLOY_USER bash -c "cd $DEPLOY_DIR/app && source $DEPLOY_DIR/venv/bin/activate && python -c 'from src.data.data_storage import DatabaseManager; DatabaseManager().test_connection()'"; then
        info "✓ Database connection successful"
    else
        error "✗ Database connection failed"
    fi
    
    log "Health check completed"
}

# Print deployment summary
print_summary() {
    log "Deployment Summary"
    echo "===================="
    echo "Application: $APP_NAME v$APP_VERSION"
    echo "Deploy Directory: $DEPLOY_DIR"
    echo "Deploy User: $DEPLOY_USER"
    echo "Service Status: $(sudo systemctl is-active market-research.service)"
    echo "Timer Status: $(sudo systemctl is-active market-research-timer.timer)"
    echo "Log File: $LOG_FILE"
    echo "===================="
    
    info "Deployment completed successfully!"
    info "You can monitor the service with: sudo systemctl status market-research.service"
    info "View logs with: sudo journalctl -u market-research.service -f"
}

# Main deployment function
main() {
    log "Starting Market Research System v$APP_VERSION deployment..."
    
    # Create log directory
    sudo mkdir -p /var/log/market-research
    sudo touch $LOG_FILE
    
    # Run deployment steps
    check_root
    check_system
    setup_user_and_dirs
    install_dependencies
    deploy_application
    configure_services
    configure_logging
    setup_database
    configure_firewall
    initial_data_collection
    start_services
    health_check
    print_summary
}

# Handle command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    stop)
        log "Stopping services..."
        sudo systemctl stop market-research.service
        sudo systemctl stop market-research-timer.timer
        info "Services stopped"
        ;;
    start)
        log "Starting services..."
        sudo systemctl start market-research.service
        sudo systemctl start market-research-timer.timer
        info "Services started"
        ;;
    restart)
        log "Restarting services..."
        sudo systemctl restart market-research.service
        sudo systemctl restart market-research-timer.timer
        info "Services restarted"
        ;;
    status)
        echo "Service Status:"
        sudo systemctl status market-research.service
        echo "Timer Status:"
        sudo systemctl status market-research-timer.timer
        ;;
    health)
        health_check
        ;;
    *)
        echo "Usage: $0 {deploy|start|stop|restart|status|health}"
        echo "  deploy  - Full deployment (default)"
        echo "  start   - Start services"
        echo "  stop    - Stop services"
        echo "  restart - Restart services"
        echo "  status  - Show service status"
        echo "  health  - Run health check"
        exit 1
        ;;
esac