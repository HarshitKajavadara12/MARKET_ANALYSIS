/**
 * Real-time Updates JavaScript - Indian Market Research Platform
 * Handles WebSocket connections, data updates, and UI refreshes
 */

class RealTimeUpdater {
    constructor() {
        this.wsConnection = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.updateInterval = 5000; // 5 seconds
        this.isConnected = false;
        this.subscriptions = new Set();
        this.dataCache = new Map();
        this.updateCallbacks = new Map();
        this.lastUpdateTime = new Date();
        
        // Market hours (IST)
        this.marketOpen = { hour: 9, minute: 15 };
        this.marketClose = { hour: 15, minute: 30 };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.startPolling();
        this.initializeWebSocket();
        this.setupMarketHoursCheck();
    }
    
    setupEventListeners() {
        // Page visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });
        
        // Window focus/blur
        window.addEventListener('focus', () => this.resumeUpdates());
        window.addEventListener('blur', () => this.pauseUpdates());
        
        // Network status
        window.addEventListener('online', () => this.handleNetworkOnline());
        window.addEventListener('offline', () => this.handleNetworkOffline());
    }
    
    initializeWebSocket() {
        try {
            // In a real implementation, this would connect to your WebSocket server
            // For demo purposes, we'll simulate WebSocket behavior
            this.simulateWebSocketConnection();
        } catch (error) {
            console.error('WebSocket initialization failed:', error);
            this.fallbackToPolling();
        }
    }
    
    simulateWebSocketConnection() {
        // Simulate WebSocket connection for demo
        setTimeout(() => {
            this.isConnected = true;
            this.onConnectionOpen();
            
            // Simulate periodic data updates
            setInterval(() => {
                if (this.isConnected && this.isMarketOpen()) {
                    this.simulateDataUpdate();
                }
            }, this.updateInterval);
        }, 1000);
    }
    
    onConnectionOpen() {
        console.log('Real-time connection established');
        this.reconnectAttempts = 0;
        this.updateConnectionStatus(true);
        
        // Subscribe to default data streams
        this.subscribeToIndices();
        this.subscribeToWatchlist();
        this.subscribeToMarketBreadth();
    }
    
    onConnectionClose() {
        console.log('Real-time connection closed');
        this.isConnected = false;
        this.updateConnectionStatus(false);
        this.attemptReconnect();
    }
    
    onConnectionError(error) {
        console.error('Real-time connection error:', error);
        this.updateConnectionStatus(false);
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);
            
            setTimeout(() => {
                this.initializeWebSocket();
            }, delay);
        } else {
            console.error('Max reconnection attempts reached. Falling back to polling.');
            this.fallbackToPolling();
        }
    }
    
    fallbackToPolling() {
        console.log('Using polling for data updates');
        this.startPolling();
    }
    
    startPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        this.pollingInterval = setInterval(() => {
            if (this.isMarketOpen()) {
                this.fetchMarketData();
            }
        }, this.updateInterval);
    }
    
    pauseUpdates() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
    }
    
    resumeUpdates() {
        if (!this.isConnected) {
            this.startPolling();
        }
    }
    
    subscribeToIndices() {
        const indices = ['NIFTY50', 'SENSEX', 'BANKNIFTY', 'NIFTYIT'];
        indices.forEach(index => {
            this.subscriptions.add(`index:${index}`);
        });
    }
    
    subscribeToWatchlist() {
        const watchlist = this.getWatchlistSymbols();
        watchlist.forEach(symbol => {
            this.subscriptions.add(`stock:${symbol}`);
        });
    }
    
    subscribeToMarketBreadth() {
        this.subscriptions.add('market:breadth');
        this.subscriptions.add('market:fii_dii');
        this.subscriptions.add('market:volume');
    }
    
    getWatchlistSymbols() {
        const watchlistItems = document.querySelectorAll('.watchlist-item');
        return Array.from(watchlistItems).map(item => item.dataset.symbol).filter(Boolean);
    }
    
    simulateDataUpdate() {
        // Simulate real-time data updates
        const updates = {
            indices: this.generateIndicesData(),
            stocks: this.generateStockData(),
            marketBreadth: this.generateMarketBreadthData(),
            fiiDii: this.generateFiiDiiData(),
            alerts: this.generateAlerts()
        };
        
        this.processDataUpdates(updates);
    }
    
    generateIndicesData() {
        const indices = {
            'NIFTY50': { current: 19500, change: 0 },
            'SENSEX': { current: 65450, change: 0 },
            'BANKNIFTY': { current: 44250, change: 0 },
            'NIFTYIT': { current: 30125, change: 0 }
        };
        
        // Add random fluctuations
        Object.keys(indices).forEach(key => {
            const data = indices[key];
            const volatility = data.current * 0.001; // 0.1% volatility
            const change = (Math.random() - 0.5) * volatility * 2;
            
            data.current += change;
            data.change = change;
            data.changePercent = (change / (data.current - change)) * 100;
            data.timestamp = new Date();
        });
        
        return indices;
    }
    
    generateStockData() {
        const stocks = {
            'RELIANCE': { current: 2485, change: 0 },
            'TCS': { current: 3650, change: 0 },
            'INFY': { current: 1420, change: 0 },
            'HDFC': { current: 1650, change: 0 }
        };
        
        Object.keys(stocks).forEach(symbol => {
            const data = stocks[symbol];
            const volatility = data.current * 0.002; // 0.2% volatility
            const change = (Math.random() - 0.5) * volatility * 2;
            
            data.current += change;
            data.change = change;
            data.changePercent = (change / (data.current - change)) * 100;
            data.volume = Math.floor(Math.random() * 1000000) + 500000;
            data.timestamp = new Date();
        });
        
        return stocks;
    }
    
    generateMarketBreadthData() {
        const advances = Math.floor(Math.random() * 500) + 1000;
        const declines = Math.floor(Math.random() * 500) + 800;
        const unchanged = Math.floor(Math.random() * 100) + 100;
        
        return {
            advances,
            declines,
            unchanged,
            advanceDeclineRatio: advances / declines,
            timestamp: new Date()
        };
    }
    
    generateFiiDiiData() {
        return {
            fii: (Math.random() - 0.6) * 5000, // Bias towards outflow
            dii: (Math.random() - 0.3) * 3000, // Bias towards inflow
            timestamp: new Date()
        };
    }
    
    generateAlerts() {
        const alerts = [];
        
        // Random chance of generating alerts
        if (Math.random() < 0.1) { // 10% chance
            const symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC'];
            const alertTypes = ['price_breach', 'volume_spike', 'technical_signal'];
            
            alerts.push({
                id: Date.now(),
                symbol: symbols[Math.floor(Math.random() * symbols.length)],
                type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
                message: 'Alert triggered',
                priority: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
                timestamp: new Date()
            });
        }
        
        return alerts;
    }
    
    processDataUpdates(updates) {
        try {
            // Update indices
            if (updates.indices) {
                this.updateIndices(updates.indices);
            }
            
            // Update stocks
            if (updates.stocks) {
                this.updateStocks(updates.stocks);
            }
            
            // Update market breadth
            if (updates.marketBreadth) {
                this.updateMarketBreadth(updates.marketBreadth);
            }
            
            // Update FII/DII data
            if (updates.fiiDii) {
                this.updateFiiDii(updates.fiiDii);
            }
            
            // Process alerts
            if (updates.alerts && updates.alerts.length > 0) {
                this.processAlerts(updates.alerts);
            }
            
            // Update last update time
            this.lastUpdateTime = new Date();
            this.updateLastUpdateDisplay();
            
        } catch (error) {
            console.error('Error processing data updates:', error);
        }
    }
    
    updateIndices(indicesData) {
        Object.keys(indicesData).forEach(indexKey => {
            const data = indicesData[indexKey];
            const element = this.getIndexElement(indexKey);
            
            if (element) {
                this.updateIndexDisplay(element, data);
                this.animateChange(element, data.change);
            }
        });
    }
    
    updateStocks(stocksData) {
        Object.keys(stocksData).forEach(symbol => {
            const data = stocksData[symbol];
            const element = this.getStockElement(symbol);
            
            if (element) {
                this.updateStockDisplay(element, data);
                this.animateChange(element, data.change);
            }
        });
    }
    
    updateMarketBreadth(breadthData) {
        const container = document.querySelector('.market-breadth');
        if (!container) return;
        
        // Update advances
        const advancesElement = container.querySelector('.breadth-value.positive');
        if (advancesElement) {
            advancesElement.textContent = breadthData.advances.toLocaleString();
        }
        
        // Update declines
        const declinesElement = container.querySelector('.breadth-value.negative');
        if (declinesElement) {
            declinesElement.textContent = breadthData.declines.toLocaleString();
        }
        
        // Update unchanged
        const unchangedElement = container.querySelector('.breadth-value.neutral');
        if (unchangedElement) {
            unchangedElement.textContent = breadthData.unchanged.toLocaleString();
        }
        
        // Update A/D ratio bar
        const ratioBar = container.querySelector('.ratio-positive');
        const ratioText = container.querySelector('.ratio-text');
        
        if (ratioBar && ratioText) {
            const percentage = (breadthData.advances / (breadthData.advances + breadthData.declines)) * 100;
            ratioBar.style.width = `${percentage}%`;
            ratioText.textContent = `A/D Ratio: ${breadthData.advanceDeclineRatio.toFixed(2)}`;
        }
    }
    
    updateFiiDii(fiiDiiData) {
        // Update FII flow
        const fiiElement = document.querySelector('.metric-card');
        if (fiiElement) {
            const valueElement = fiiElement.querySelector('.metric-value');
            const changeElement = fiiElement.querySelector('.metric-change');
            
            if (valueElement && changeElement) {
                const fiiValue = fiiDiiData.fii;
                valueElement.textContent = `₹${Math.abs(fiiValue).toFixed(0)} Cr`;
                
                changeElement.textContent = fiiValue >= 0 ? 'Inflow' : 'Outflow';
                changeElement.className = `metric-change ${fiiValue >= 0 ? 'positive' : 'negative'}`;
            }
        }
    }
    
    getIndexElement(indexKey) {
        const mapping = {
            'NIFTY50': 'nifty-50',
            'SENSEX': 'sensex',
            'BANKNIFTY': 'bank-nifty',
            'NIFTYIT': 'nifty-it'
        };
        
        const elementId = mapping[indexKey];
        return elementId ? document.getElementById(elementId) : null;
    }
    
    getStockElement(symbol) {
        return document.querySelector(`[data-symbol="${symbol}"]`);
    }
    
    updateIndexDisplay(element, data) {
        const valueElement = element.querySelector('.index-value');
        const changeElement = element.querySelector('.index-change');
        
        if (valueElement) {
            valueElement.textContent = data.current.toFixed(2);
        }
        
        if (changeElement) {
            const changeText = `${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)} (${data.changePercent >= 0 ? '+' : ''}${data.changePercent.toFixed(2)}%)`;
            changeElement.textContent = changeText;
            changeElement.className = `index-change ${data.change >= 0 ? 'positive' : 'negative'}`;
        }
    }
    
    updateStockDisplay(element, data) {
        const priceElement = element.querySelector('.stock-price');
        const changeElement = element.querySelector('.stock-change');
        
        if (priceElement) {
            priceElement.textContent = `₹${data.current.toFixed(2)}`;
        }
        
        if (changeElement) {
            const changeText = `${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)} (${data.changePercent >= 0 ? '+' : ''}${data.changePercent.toFixed(2)}%)`;
            changeElement.textContent = changeText;
            changeElement.className = `stock-change ${data.change >= 0 ? 'positive' : 'negative'}`;
        }
    }
    
    animateChange(element, change) {
        // Add flash animation for price changes
        const flashClass = change >= 0 ? 'flash-green' : 'flash-red';
        element.classList.add(flashClass);
        
        setTimeout(() => {
            element.classList.remove(flashClass);
        }, 500);
    }
    
    processAlerts(alerts) {
        alerts.forEach(alert => {
            this.showAlert(alert);
            this.updateAlertsPanel(alert);
        });
    }
    
    showAlert(alert) {
        // Show toast notification
        this.showToast(alert.message, alert.priority);
        
        // Update alerts badge
        this.updateAlertsBadge();
    }
    
    showToast(message, priority = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${priority}`;
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas fa-bell"></i>
                <span>${message}</span>
                <button class="toast-close">&times;</button>
            </div>
        `;
        
        // Add to page
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
        
        // Close button functionality
        const closeBtn = toast.querySelector('.toast-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            });
        }
    }
    
    updateAlertsPanel(alert) {
        const alertsContainer = document.querySelector('.alerts-content');
        if (!alertsContainer) return;
        
        const alertElement = document.createElement('div');
        alertElement.className = `alert-item ${alert.priority}-priority`;
        alertElement.innerHTML = `
            <div class="alert-icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="alert-content">
                <div class="alert-title">${alert.symbol} - ${alert.message}</div>
                <div class="alert-time">Just now</div>
            </div>
        `;
        
        // Add to top of alerts list
        alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);
        
        // Remove old alerts if too many
        const alertItems = alertsContainer.querySelectorAll('.alert-item');
        if (alertItems.length > 5) {
            alertsContainer.removeChild(alertItems[alertItems.length - 1]);
        }
    }
    
    updateAlertsBadge() {
        const badge = document.querySelector('.alerts-panel .badge');
        if (badge) {
            const currentCount = parseInt(badge.textContent) || 0;
            badge.textContent = currentCount + 1;
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('market-status');
        if (statusElement) {
            if (connected) {
                statusElement.className = 'badge bg-success';
                statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Connected';
            } else {
                statusElement.className = 'badge bg-danger';
                statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Disconnected';
            }
        }
    }
    
    updateLastUpdateDisplay() {
        const updateElement = document.querySelector('.last-update');
        if (updateElement) {
            updateElement.textContent = `Last updated: ${this.lastUpdateTime.toLocaleTimeString()}`;
        }
    }
    
    isMarketOpen() {
        const now = new Date();
        const istTime = new Date(now.toLocaleString("en-US", {timeZone: "Asia/Kolkata"}));
        
        const currentHour = istTime.getHours();
        const currentMinute = istTime.getMinutes();
        const currentDay = istTime.getDay(); // 0 = Sunday, 6 = Saturday
        
        // Check if it's a weekend
        if (currentDay === 0 || currentDay === 6) {
            return false;
        }
        
        // Check if within market hours
        const marketStartMinutes = this.marketOpen.hour * 60 + this.marketOpen.minute;
        const marketEndMinutes = this.marketClose.hour * 60 + this.marketClose.minute;
        const currentMinutes = currentHour * 60 + currentMinute;
        
        return currentMinutes >= marketStartMinutes && currentMinutes <= marketEndMinutes;
    }
    
    setupMarketHoursCheck() {
        // Check market status every minute
        setInterval(() => {
            const isOpen = this.isMarketOpen();
            this.updateMarketStatus(isOpen);
        }, 60000);
        
        // Initial check
        this.updateMarketStatus(this.isMarketOpen());
    }
    
    updateMarketStatus(isOpen) {
        const statusElement = document.getElementById('market-status');
        if (statusElement) {
            if (isOpen) {
                statusElement.className = 'badge bg-success';
                statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Market Open';
            } else {
                statusElement.className = 'badge bg-secondary';
                statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Market Closed';
            }
        }
    }
    
    handleNetworkOnline() {
        console.log('Network connection restored');
        this.resumeUpdates();
        if (!this.isConnected) {
            this.initializeWebSocket();
        }
    }
    
    handleNetworkOffline() {
        console.log('Network connection lost');
        this.pauseUpdates();
        this.updateConnectionStatus(false);
    }
    
    fetchMarketData() {
        // Fallback method for fetching data via HTTP
        // In a real implementation, this would make API calls
        console.log('Fetching market data via HTTP...');
        this.simulateDataUpdate();
    }
    
    subscribe(channel, callback) {
        this.subscriptions.add(channel);
        if (callback) {
            this.updateCallbacks.set(channel, callback);
        }
    }
    
    unsubscribe(channel) {
        this.subscriptions.delete(channel);
        this.updateCallbacks.delete(channel);
    }
    
    destroy() {
        // Cleanup method
        if (this.wsConnection) {
            this.wsConnection.close();
        }
        
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        this.subscriptions.clear();
        this.updateCallbacks.clear();
        this.dataCache.clear();
    }
}

// CSS for toast notifications and animations
const toastStyles = `
<style>
.toast-container {
    position: fixed;
    top: 80px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.toast {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-tertiary));
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    min-width: 300px;
    max-width: 400px;
    animation: slideInRight 0.3s ease-out;
    color: var(--text-primary);
}

.toast-info {
    border-left: 4px solid var(--info-color);
}

.toast-low {
    border-left: 4px solid var(--bullish-color);
}

.toast-medium {
    border-left: 4px solid var(--neutral-color);
}

.toast-high {
    border-left: 4px solid var(--bearish-color);
}

.toast-content {
    display: flex;
    align-items: center;
    gap: 10px;
}

.toast-close {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 18px;
    cursor: pointer;
    margin-left: auto;
    padding: 0;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.toast-close:hover {
    color: var(--text-primary);
}

.flash-green {
    animation: flashGreen 0.5s ease-out;
}

.flash-red {
    animation: flashRed 0.5s ease-out;
}

@keyframes flashGreen {
    0% { background-color: transparent; }
    50% { background-color: rgba(0, 212, 170, 0.3); }
    100% { background-color: transparent; }
}

@keyframes flashRed {
    0% { background-color: transparent; }
    50% { background-color: rgba(255, 71, 87, 0.3); }
    100% { background-color: transparent; }
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(100%);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}
</style>
`;

// Add styles to document
document.head.insertAdjacentHTML('beforeend', toastStyles);

// Global instance
let realTimeUpdater;

// Initialize when DOM is ready
function startRealTimeUpdates() {
    if (!realTimeUpdater) {
        realTimeUpdater = new RealTimeUpdater();
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (realTimeUpdater) {
        realTimeUpdater.destroy();
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RealTimeUpdater;
}

// Make available globally
window.RealTimeUpdater = RealTimeUpdater;
window.startRealTimeUpdates = startRealTimeUpdates;