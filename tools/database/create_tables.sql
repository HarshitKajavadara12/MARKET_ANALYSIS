-- Market Research System v1.0 Database Schema (2022)
-- Compatible with SQLite, MySQL, and PostgreSQL
-- Focus: Indian Stock Market

-- =============================================================================
-- STOCK DATA TABLES
-- =============================================================================

-- Stock symbols master table
CREATE TABLE IF NOT EXISTS stock_symbols (
    symbol VARCHAR(20) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    market_cap DECIMAL(15,2),
    listing_date DATE,
    exchange VARCHAR(10) DEFAULT 'NSE',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily stock price data
CREATE TABLE IF NOT EXISTS daily_stock_prices (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10,2) NOT NULL,
    high_price DECIMAL(10,2) NOT NULL,
    low_price DECIMAL(10,2) NOT NULL,
    close_price DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES stock_symbols(symbol),
    UNIQUE(symbol, date)
);

-- Intraday stock price data (for real-time analysis)
CREATE TABLE IF NOT EXISTS intraday_stock_prices (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    volume INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES stock_symbols(symbol),
    UNIQUE(symbol, timestamp)
);

-- =============================================================================
-- TECHNICAL INDICATORS TABLES
-- =============================================================================

-- Technical indicators storage
CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value DECIMAL(15,6),
    period_length INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES stock_symbols(symbol),
    UNIQUE(symbol, date, indicator_name, period_length)
);

-- Moving averages (optimized separate table)
CREATE TABLE IF NOT EXISTS moving_averages (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    sma_10 DECIMAL(10,2),
    sma_20 DECIMAL(10,2),
    sma_50 DECIMAL(10,2),
    sma_200 DECIMAL(10,2),
    ema_10 DECIMAL(10,2),
    ema_20 DECIMAL(10,2),
    ema_50 DECIMAL(10,2),
    ema_200 DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES stock_symbols(symbol),
    UNIQUE(symbol, date)
);

-- =============================================================================
-- MARKET INDICES TABLES
-- =============================================================================

-- Indian market indices
CREATE TABLE IF NOT EXISTS market_indices (
    id INTEGER PRIMARY KEY,
    index_name VARCHAR(50) NOT NULL,
    index_symbol VARCHAR(20) UNIQUE NOT NULL,
    description TEXT,
    base_value DECIMAL(10,2),
    base_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily index values
CREATE TABLE IF NOT EXISTS daily_index_values (
    id INTEGER PRIMARY KEY,
    index_symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_value DECIMAL(12,2) NOT NULL,
    high_value DECIMAL(12,2) NOT NULL,
    low_value DECIMAL(12,2) NOT NULL,
    close_value DECIMAL(12,2) NOT NULL,
    volume BIGINT,
    change_value DECIMAL(10,2),
    change_percent DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (index_symbol) REFERENCES market_indices(index_symbol),
    UNIQUE(index_symbol, date)
);

-- =============================================================================
-- ECONOMIC DATA TABLES
-- =============================================================================

-- Economic indicators
CREATE TABLE IF NOT EXISTS economic_indicators (
    id INTEGER PRIMARY KEY,
    indicator_name VARCHAR(100) NOT NULL,
    indicator_code VARCHAR(50) UNIQUE,
    category VARCHAR(50),
    unit VARCHAR(20),
    frequency VARCHAR(20), -- Daily, Weekly, Monthly, Quarterly, Yearly
    source VARCHAR(100),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Economic data values
CREATE TABLE IF NOT EXISTS economic_data (
    id INTEGER PRIMARY KEY,
    indicator_code VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    value DECIMAL(15,4),
    period_type VARCHAR(20), -- actual, forecast, revised
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (indicator_code) REFERENCES economic_indicators(indicator_code),
    UNIQUE(indicator_code, date, period_type)
);

-- =============================================================================
-- NEWS AND SENTIMENT DATA
-- =============================================================================

-- News articles
CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT,
    source VARCHAR(100),
    author VARCHAR(255),
    published_at TIMESTAMP,
    url VARCHAR(1000),
    sentiment_score DECIMAL(3,2), -- -1 to 1
    relevance_score DECIMAL(3,2), -- 0 to 1
    category VARCHAR(50),
    symbols_mentioned TEXT, -- Comma-separated list
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Social media sentiment (aggregated)
CREATE TABLE IF NOT EXISTS social_sentiment (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20),
    date DATE NOT NULL,
    platform VARCHAR(50), -- Twitter, Reddit, etc.
    sentiment_score DECIMAL(3,2),
    mention_count INTEGER,
    positive_mentions INTEGER,
    negative_mentions INTEGER,
    neutral_mentions INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES stock_symbols(symbol),
    UNIQUE(symbol, date, platform)
);

-- =============================================================================
-- ANALYSIS AND REPORTS
-- =============================================================================

-- Analysis results storage
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY,
    analysis_type VARCHAR(100) NOT NULL,
    symbol VARCHAR(20),
    analysis_date DATE NOT NULL,
    result_data TEXT, -- JSON format for flexibility
    confidence_score DECIMAL(3,2),
    recommendation VARCHAR(20), -- BUY, SELL, HOLD
    target_price DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    time_horizon VARCHAR(20), -- Short, Medium, Long term
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES stock_symbols(symbol)
);

-- Report generation log
CREATE TABLE IF NOT EXISTS report_logs (
    id INTEGER PRIMARY KEY,
    report_type VARCHAR(50) NOT NULL,
    report_date DATE NOT NULL,
    file_path VARCHAR(500),
    generation_time DECIMAL(10,3), -- seconds
    status VARCHAR(20) DEFAULT 'SUCCESS',
    error_message TEXT,
    parameters TEXT, -- JSON format
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- SYSTEM MONITORING
-- =============================================================================

-- Data collection status
CREATE TABLE IF NOT EXISTS data_collection_status (
    id INTEGER PRIMARY KEY,
    data_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    last_update TIMESTAMP,
    next_update TIMESTAMP,
    status VARCHAR(20) DEFAULT 'PENDING',
    records_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System performance metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4),
    metric_unit VARCHAR(20),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category VARCHAR(50)
);

-- =============================================================================
-- USER AND CLIENT MANAGEMENT
-- =============================================================================

-- Client information
CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY,
    client_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    subscription_type VARCHAR(50), -- Basic, Premium, Enterprise
    subscription_start DATE,
    subscription_end DATE,
    is_active BOOLEAN DEFAULT TRUE,
    preferences TEXT, -- JSON format
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Report delivery log
CREATE TABLE IF NOT EXISTS report_deliveries (
    id INTEGER PRIMARY KEY,
    client_id INTEGER,
    report_type VARCHAR(50),
    delivery_method VARCHAR(20), -- Email, API, Dashboard
    delivery_time TIMESTAMP,
    status VARCHAR(20), -- SENT, FAILED, PENDING
    file_path VARCHAR(500),
    error_message TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(id)
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =============================================================================

-- Stock prices indexes
CREATE INDEX IF NOT EXISTS idx_daily_stock_prices_symbol_date ON daily_stock_prices(symbol, date);
CREATE INDEX IF NOT EXISTS idx_daily_stock_prices_date ON daily_stock_prices(date);
CREATE INDEX IF NOT EXISTS idx_intraday_stock_prices_symbol_timestamp ON intraday_stock_prices(symbol, timestamp);

-- Technical indicators indexes
CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_date ON technical_indicators(symbol, date);
CREATE INDEX IF NOT EXISTS idx_moving_averages_symbol_date ON moving_averages(symbol, date);

-- Index values indexes
CREATE INDEX IF NOT EXISTS idx_daily_index_values_symbol_date ON daily_index_values(index_symbol, date);

-- Economic data indexes
CREATE INDEX IF NOT EXISTS idx_economic_data_code_date ON economic_data(indicator_code, date);

-- News and sentiment indexes
CREATE INDEX IF NOT EXISTS idx_news_articles_published_at ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_social_sentiment_symbol_date ON social_sentiment(symbol, date);

-- Analysis results indexes
CREATE INDEX IF NOT EXISTS idx_analysis_results_symbol_date ON analysis_results(symbol, analysis_date);

-- =============================================================================
-- INITIAL DATA INSERTS
-- =============================================================================

-- Insert major Indian market indices
INSERT OR IGNORE INTO market_indices (index_name, index_symbol, description, base_value, base_date) VALUES
('NIFTY 50', '^NSEI', 'NSE NIFTY 50 Index', 1000.00, '1995-11-03'),
('NIFTY Bank', '^NSEBANK', 'NSE Bank Index', 1000.00, '2000-01-01'),
('SENSEX', '^BSESN', 'BSE SENSEX Index', 100.00, '1979-04-01'),
('NIFTY IT', '^NSEIT', 'NSE IT Index', 1000.00, '2001-01-01'),
('NIFTY Auto', '^NSEAUTO', 'NSE Auto Index', 1000.00, '2001-01-01');

-- Insert major Indian stocks (Top 50 by market cap as of 2022)
INSERT OR IGNORE INTO stock_symbols (symbol, company_name, sector, exchange) VALUES
('RELIANCE.NS', 'Reliance Industries Limited', 'Energy', 'NSE'),
('TCS.NS', 'Tata Consultancy Services Limited', 'Information Technology', 'NSE'),
('HDFCBANK.NS', 'HDFC Bank Limited', 'Financial Services', 'NSE'),
('INFY.NS', 'Infosys Limited', 'Information Technology', 'NSE'),
('ICICIBANK.NS', 'ICICI Bank Limited', 'Financial Services', 'NSE'),
('HINDUNILVR.NS', 'Hindustan Unilever Limited', 'FMCG', 'NSE'),
('ITC.NS', 'ITC Limited', 'FMCG', 'NSE'),
('SBIN.NS', 'State Bank of India', 'Financial Services', 'NSE'),
('BHARTIARTL.NS', 'Bharti Airtel Limited', 'Telecommunications', 'NSE'),
('KOTAKBANK.NS', 'Kotak Mahindra Bank Limited', 'Financial Services', 'NSE'),
('LT.NS', 'Larsen & Toubro Limited', 'Construction', 'NSE'),
('HCLTECH.NS', 'HCL Technologies Limited', 'Information Technology', 'NSE'),
('ASIANPAINT.NS', 'Asian Paints Limited', 'Consumer Durables', 'NSE'),
('MARUTI.NS', 'Maruti Suzuki India Limited', 'Automobile', 'NSE'),
('SUNPHARMA.NS', 'Sun Pharmaceutical Industries Limited', 'Pharmaceuticals', 'NSE'),
('TITAN.NS', 'Titan Company Limited', 'Consumer Durables', 'NSE'),
('ULTRACEMCO.NS', 'UltraTech Cement Limited', 'Cement', 'NSE'),
('TECHM.NS', 'Tech Mahindra Limited', 'Information Technology', 'NSE'),
('ONGC.NS', 'Oil and Natural Gas Corporation Limited', 'Energy', 'NSE'),
('WIPRO.NS', 'Wipro Limited', 'Information Technology', 'NSE');

-- Insert economic indicators
INSERT OR IGNORE INTO economic_indicators (indicator_name, indicator_code, category, unit, frequency, source) VALUES
('Consumer Price Index', 'CPI_IN', 'Inflation', 'Index', 'Monthly', 'RBI'),
('Gross Domestic Product', 'GDP_IN', 'Growth', 'Percentage', 'Quarterly', 'NSO'),
('Repo Rate', 'REPO_RATE', 'Monetary Policy', 'Percentage', 'Monthly', 'RBI'),
('Wholesale Price Index', 'WPI_IN', 'Inflation', 'Index', 'Monthly', 'DPIIT'),
('Industrial Production Index', 'IIP_IN', 'Production', 'Index', 'Monthly', 'NSO'),
('Foreign Exchange Reserves', 'FOREX_RESERVES', 'Currency', 'USD Billion', 'Weekly', 'RBI');

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Latest stock prices view
CREATE VIEW IF NOT EXISTS latest_stock_prices AS
SELECT 
    s.symbol,
    s.company_name,
    s.sector,
    dsp.date,
    dsp.close_price,
    dsp.volume,
    dsp.open_price,
    dsp.high_price,
    dsp.low_price
FROM stock_symbols s
JOIN daily_stock_prices dsp ON s.symbol = dsp.symbol
WHERE dsp.date = (
    SELECT MAX(date) 
    FROM daily_stock_prices dsp2 
    WHERE dsp2.symbol = s.symbol
);

-- Market overview view
CREATE VIEW IF NOT EXISTS market_overview AS
SELECT 
    mi.index_name,
    mi.index_symbol,
    div.close_value,
    div.change_value,
    div.change_percent,
    div.volume,
    div.date
FROM market_indices mi
JOIN daily_index_values div ON mi.index_symbol = div.index_symbol
WHERE div.date = (
    SELECT MAX(date) 
    FROM daily_index_values div2 
    WHERE div2.index_symbol = mi.index_symbol
);

-- =============================================================================
-- TRIGGERS FOR DATA INTEGRITY
-- =============================================================================

-- Update timestamp trigger for stock_symbols
CREATE TRIGGER IF NOT EXISTS update_stock_symbols_timestamp
    AFTER UPDATE ON stock_symbols
    FOR EACH ROW
BEGIN
    UPDATE stock_symbols SET updated_at = CURRENT_TIMESTAMP WHERE symbol = NEW.symbol;
END;

-- Update data collection status trigger
CREATE TRIGGER IF NOT EXISTS update_data_collection_status
    AFTER INSERT ON daily_stock_prices
    FOR EACH ROW
BEGIN
    INSERT OR REPLACE INTO data_collection_status 
    (data_type, symbol, last_update, status, records_count, updated_at)
    VALUES 
    ('daily_prices', NEW.symbol, CURRENT_TIMESTAMP, 'SUCCESS', 
     (SELECT COUNT(*) FROM daily_stock_prices WHERE symbol = NEW.symbol),
     CURRENT_TIMESTAMP);
END;