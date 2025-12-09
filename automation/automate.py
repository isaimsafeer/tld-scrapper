from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from typing import Dict, Any, List
import logging
from datetime import datetime
import time
import re
import json
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def get_db_connection(db_path: str = "domain_pricing.db"):
    """
    Context manager for database connections
    
    Args:
        db_path: Path to SQLite database file
        
    Yields:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database(db_path: str = "domain_pricing.db"):
    """
    Initialize SQLite database with required tables
    
    Args:
        db_path: Path to SQLite database file
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Create raw_data table for unprocessed scraped data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                row_number INTEGER,
                domain TEXT,
                column_name TEXT,
                cell_value TEXT,
                scraped_at TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create structured_pricing table for processed pricing data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS structured_pricing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                tld TEXT NOT NULL,
                registrar TEXT NOT NULL,
                operation TEXT NOT NULL,
                price REAL,
                currency TEXT DEFAULT 'USD',
                years INTEGER DEFAULT 1,
                pricing_notes TEXT,
                promo_code TEXT,
                scraped_at TEXT,
                source_url TEXT,
                column_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(task_id, tld, registrar, operation, column_name)
            )
        """)
        
        # Create scraping_tasks table to track scraping jobs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scraping_tasks (
                task_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                status TEXT NOT NULL,
                page_title TEXT,
                rows_extracted INTEGER DEFAULT 0,
                structured_records INTEGER DEFAULT 0,
                columns INTEGER DEFAULT 0,
                headers TEXT,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_structured_tld 
            ON structured_pricing(tld)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_structured_registrar 
            ON structured_pricing(registrar)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_structured_operation 
            ON structured_pricing(operation)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_raw_task 
            ON raw_data(task_id)
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")


def setup_driver(headless: bool = False) -> webdriver.Chrome:
    """
    Setup Chrome WebDriver with options
    
    Args:
        headless: Run browser in headless mode
        
    Returns:
        Configured Chrome WebDriver instance
    """
    chrome_options = Options()
    
    logger.info(f"Setting up driver with headless={headless}")
    
    if headless:
        chrome_options.add_argument("--headless")
        logger.info("Running in HEADLESS mode")
    else:
        logger.info("Running in VISIBLE mode (browser should open)")
    
    # Additional options for stability
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_experimental_option("detach", True)
    
    # Initialize driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    return driver


def extract_registrar_from_url(url: str) -> str:
    """
    Extract registrar name from URL
    
    Args:
        url: Full URL of the page
        
    Returns:
        Registrar name (domain name without TLD)
    """
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if match:
        domain = match.group(1)
        parts = domain.split('.')
        if len(parts) >= 2:
            return parts[0]
        return domain
    return "unknown"


def parse_price_data(cell_value: str) -> Dict[str, Any]:
    """
    Parse price information from cell value
    
    Args:
        cell_value: Raw cell value containing registrar, promo, and price
        
    Returns:
        Dictionary with parsed price information
    """
    result = {
        "registrar": "",
        "price": None,
        "currency": "USD",
        "pricing_notes": "",
        "promo_code": ""
    }
    
    if not cell_value or cell_value.strip() == "":
        return result
    
    logger.debug(f"Parsing cell value: {repr(cell_value[:100])}")
    
    lines = cell_value.strip().split('\n')
    
    # First line is usually the registrar
    if lines:
        result["registrar"] = lines[0].strip().lower()
        logger.debug(f"Extracted registrar: {result['registrar']}")
    
    # Look for promo code
    promo_match = re.search(r'Promo code\s+(\S+)', cell_value, re.IGNORECASE)
    if promo_match:
        result["promo_code"] = promo_match.group(1)
        result["pricing_notes"] = f"promo code: {promo_match.group(1)}"
        logger.debug(f"Found promo code: {result['promo_code']}")
    
    # Look for price with currency symbol
    price_patterns = [
        (r'\$(\d+\.?\d*)', 'USD'),
        (r'£➜\s*\$(\d+\.?\d*)', 'USD'),
        (r'£(\d+\.?\d*)', 'GBP'),
        (r'€(\d+\.?\d*)', 'EUR'),
    ]
    
    for pattern, currency in price_patterns:
        price_match = re.search(pattern, cell_value)
        if price_match:
            result["price"] = float(price_match.group(1))
            result["currency"] = currency
            logger.debug(f"Found price: {result['price']} {currency}")
            break
    
    if result["price"] is None:
        logger.warning(f"Could not extract price from: {repr(cell_value[:100])}")
    
    # Check for special notes
    if "registration" in cell_value.lower():
        if result["pricing_notes"]:
            result["pricing_notes"] += ", registration price"
        else:
            result["pricing_notes"] = "registration price"
    
    if "renewal" in cell_value.lower() and "registration" in cell_value.lower():
        if result["pricing_notes"]:
            result["pricing_notes"] += ", includes renewal"
        else:
            result["pricing_notes"] = "includes renewal"
    
    return result


def map_column_to_operation(column_name: str) -> str:
    """
    Map column header to operation type
    
    Args:
        column_name: Column header text
        
    Returns:
        Operation type (register, renew, transfer)
    """
    column_lower = column_name.lower()
    
    if "registration" in column_lower or "register" in column_lower:
        return "register"
    elif "renewal" in column_lower or "renew" in column_lower:
        return "renew"
    elif "transfer" in column_lower:
        return "transfer"
    elif "value" in column_lower or "year" in column_lower:
        return "multi_year"
    else:
        return "unknown"


def save_raw_data_to_db(headers: List[str], rows_data: List[List[str]], 
                        task_id: str, url: str, db_path: str = "domain_pricing.db"):
    """
    Save raw scraped data to database
    
    Args:
        headers: List of column headers
        rows_data: List of row data
        task_id: Task identifier
        url: Source URL
        db_path: Path to SQLite database
    """
    scraped_at = datetime.now().isoformat()
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        for row_num, row in enumerate(rows_data, start=1):
            # First column is the domain/TLD
            domain = row[0] if row else ""
            
            # Save each cell as a separate record
            for col_idx, cell_value in enumerate(row[1:], start=1):
                if col_idx < len(headers):
                    column_name = headers[col_idx]
                    
                    cursor.execute("""
                        INSERT INTO raw_data (task_id, row_number, domain, column_name, 
                                            cell_value, scraped_at, source_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (task_id, row_num, domain, column_name, cell_value, 
                          scraped_at, url))
        
        logger.info(f"Task {task_id}: Saved {len(rows_data)} raw data rows to database")


def process_and_save_structured_data(headers: List[str], rows_data: List[List[str]], 
                                     task_id: str, url: str, 
                                     db_path: str = "domain_pricing.db") -> int:
    """
    Process raw data and save structured pricing data to database
    
    Args:
        headers: List of column headers
        rows_data: List of row data
        task_id: Task identifier
        url: Source URL
        db_path: Path to SQLite database
        
    Returns:
        Number of structured records created
    """
    scraped_at = datetime.now().isoformat()
    registrar_from_url = extract_registrar_from_url(url)
    records_count = 0
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        for row_idx, row in enumerate(rows_data, start=2):
            if not row or not any(row):
                continue
            
            # Get TLD from first or second column
            tld = ""
            tld_col_idx = -1
            
            if row[0] and str(row[0]).strip():
                tld = str(row[0]).strip()
                tld_col_idx = 0
            elif len(row) > 1 and row[1] and str(row[1]).strip():
                tld = str(row[1]).strip()
                tld_col_idx = 1
            
            if not tld or not tld.startswith('.'):
                logger.warning(f"Task {task_id}: Row {row_idx} has no valid TLD, skipping")
                continue
            
            logger.info(f"Task {task_id}: Processing row {row_idx} - TLD: {tld}")
            
            # Process each column after the TLD column
            start_col = tld_col_idx + 1
            for col_idx in range(start_col, len(row)):
                if col_idx >= len(headers):
                    break
                
                cell_value = row[col_idx]
                column_name = headers[col_idx]
                
                if not column_name or not cell_value or str(cell_value).strip() == "":
                    continue
                
                operation = map_column_to_operation(column_name)
                price_info = parse_price_data(str(cell_value))
                
                if price_info["price"] is None:
                    logger.warning(f"Task {task_id}: No price in row {row_idx}, column '{column_name}'")
                    continue
                
                # Insert structured record (ignore duplicates)
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO structured_pricing 
                        (task_id, tld, registrar, operation, price, currency, years, 
                         pricing_notes, promo_code, scraped_at, source_url, column_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task_id,
                        tld,
                        price_info["registrar"] if price_info["registrar"] else registrar_from_url,
                        operation,
                        price_info["price"],
                        price_info["currency"],
                        3 if operation == "multi_year" else 1,
                        price_info["pricing_notes"],
                        price_info["promo_code"],
                        scraped_at,
                        url,
                        column_name
                    ))
                    
                    records_count += 1
                    logger.info(f"Task {task_id}: Added record - TLD: {tld}, "
                              f"Registrar: {price_info['registrar'] or registrar_from_url}, "
                              f"Operation: {operation}, Price: {price_info['price']}")
                except sqlite3.IntegrityError as e:
                    logger.warning(f"Task {task_id}: Duplicate record skipped - {e}")
        
        logger.info(f"Task {task_id}: Saved {records_count} structured records to database")
    
    return records_count


def save_task_info(task_id: str, url: str, status: str, 
                   page_title: str = None, rows_extracted: int = 0,
                   structured_records: int = 0, columns: int = 0,
                   headers: List[str] = None, error_message: str = None,
                   db_path: str = "domain_pricing.db"):
    """
    Save or update task information in database
    
    Args:
        task_id: Task identifier
        url: Source URL
        status: Task status (started, completed, failed)
        page_title: Page title
        rows_extracted: Number of rows extracted
        structured_records: Number of structured records created
        columns: Number of columns
        headers: List of headers
        error_message: Error message if failed
        db_path: Path to SQLite database
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        headers_json = json.dumps(headers) if headers else None
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO scraping_tasks 
            (task_id, url, status, page_title, rows_extracted, structured_records, 
             columns, headers, started_at, completed_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                    COALESCE((SELECT started_at FROM scraping_tasks WHERE task_id = ?), ?),
                    ?, ?)
        """, (task_id, url, status, page_title, rows_extracted, structured_records,
              columns, headers_json, task_id, now, 
              now if status == 'completed' else None, error_message))


def export_to_json(task_id: str, db_path: str = "domain_pricing.db") -> str:
    """
    Export structured data for a task to JSON file
    
    Args:
        task_id: Task identifier
        db_path: Path to SQLite database
        
    Returns:
        Path to JSON file
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tld, registrar, operation, price, currency, years, 
                   pricing_notes, promo_code, scraped_at, source_url, column_name
            FROM structured_pricing
            WHERE task_id = ?
            ORDER BY tld, operation
        """, (task_id,))
        
        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
    
    json_filename = f"structured_data_{task_id}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Task {task_id}: Exported {len(data)} records to {json_filename}")
    return json_filename


def clear_database(db_path: str = "domain_pricing.db"):
    """
    Clear all data from the database tables
    
    Args:
        db_path: Path to SQLite database file
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        logger.info("Clearing all data from database...")
        
        # Delete all records from tables
        cursor.execute("DELETE FROM structured_pricing")
        deleted_pricing = cursor.rowcount
        
        cursor.execute("DELETE FROM raw_data")
        deleted_raw = cursor.rowcount
        
        cursor.execute("DELETE FROM scraping_tasks")
        deleted_tasks = cursor.rowcount
        
        conn.commit()
        
        logger.info(f"Database cleared: {deleted_pricing} pricing records, "
                   f"{deleted_raw} raw data records, {deleted_tasks} tasks removed")


def run_automation(task_id: str, url: str, timeout: int, headless: bool,
                   db_path: str = "domain_pricing.db", clear_before_run: bool = True) -> Dict[str, Any]:
    """
    Main automation function
    
    Args:
        task_id: Unique task identifier
        url: Target website URL
        timeout: Maximum wait time for elements
        headless: Run in headless mode
        db_path: Path to SQLite database
        clear_before_run: Clear database before running (default: True)
        
    Returns:
        Dictionary containing automation results or error information
    """
    driver = None
    
    # Initialize database
    init_database(db_path)
    
    # Clear database if requested
    if clear_before_run:
        clear_database(db_path)
    
    try:
        logger.info(f"Task {task_id}: Starting automation for {url}")
        
        # Save initial task info
        save_task_info(task_id, url, "started", db_path=db_path)
        
        # Setup driver
        driver = setup_driver(headless=headless)
        
        # Navigate to URL
        driver.get(url)
        logger.info(f"Task {task_id}: Navigated to {url}")
        
        # Wait for page to load
        wait = WebDriverWait(driver, timeout)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        page_title = driver.title
        logger.info(f"Task {task_id}: Page title: {page_title}")
        
        # Wait for table to load
        logger.info(f"Task {task_id}: Waiting for table to load...")
        table = wait.until(EC.presence_of_element_located(
            (By.XPATH, "(//table[@id='DataTables_Table_0'])[1]")
        ))
        logger.info(f"Task {task_id}: Table found!")
        
        # Extract headers
        headers = []
        try:
            header_cells = driver.find_elements(By.XPATH, 
                "//div[contains(@class,'dataTables_scrollHeadInner')]//thead//tr//th")
            headers = [cell.text.strip() for cell in header_cells if cell.text.strip()]
            
            if headers:
                headers.insert(0, "Domain")
                logger.info(f"Task {task_id}: Found {len(headers)} headers: {headers}")
        except Exception as e:
            logger.warning(f"Task {task_id}: Could not extract headers: {e}")
        
        # Extract table rows from multiple pages
        rows_data = []
        
        # Find total pages
        last_page_element = driver.find_element(By.XPATH, 
            "//div[@id='DataTables_Table_0_paginate']//li[last()-1]")
        pages_to_scrape = int(last_page_element.text)
        print(f"Total pages: {pages_to_scrape}")
        
        for page_num in range(pages_to_scrape):
            logger.info(f"Task {task_id}: Extracting data from page {page_num + 1}...")
            time.sleep(2)
            
            table_rows = driver.find_elements(By.XPATH, 
                "(//table[@id='DataTables_Table_0'])[1]//tbody//tr")
            logger.info(f"Task {task_id}: Found {len(table_rows)} rows on page {page_num + 1}")
            
            for idx, row in enumerate(table_rows):
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    row_data = [cell.text.strip() for cell in cells]
                    
                    if row_data:
                        rows_data.append(row_data)
                        logger.debug(f"Task {task_id}: Row {idx}: {row_data[:3]}...")
                except Exception as row_error:
                    logger.warning(f"Task {task_id}: Error extracting row {idx}: {row_error}")
                    continue
            
            logger.info(f"Task {task_id}: Total rows so far: {len(rows_data)}")
            
            # Click Next button if not on last page
            if page_num < pages_to_scrape - 1:
                try:
                    next_button = driver.find_element(By.XPATH, 
                        "//*[@id='DataTables_Table_0_next']/a")
                    
                    parent_li = driver.find_element(By.XPATH, 
                        "//*[@id='DataTables_Table_0_next']")
                    if "disabled" in parent_li.get_attribute("class"):
                        logger.info(f"Task {task_id}: Next button disabled")
                        break
                    
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(1)
                    next_button.click()
                    logger.info(f"Task {task_id}: Clicked Next to page {page_num + 2}")
                    time.sleep(2)
                except Exception as click_error:
                    logger.warning(f"Task {task_id}: Could not click Next: {click_error}")
                    break
        
        logger.info(f"Task {task_id}: Extracted {len(rows_data)} total rows")
        
        # Save raw data to database
        save_raw_data_to_db(headers, rows_data, task_id, url, db_path)
        
        # Process and save structured data
        structured_count = process_and_save_structured_data(
            headers, rows_data, task_id, url, db_path)
        
        # Export to JSON (optional)
        json_filename = export_to_json(task_id, db_path)
        
        # Update task info
        save_task_info(
            task_id, url, "completed", page_title, 
            len(rows_data), structured_count, len(headers), headers,
            db_path=db_path
        )
        
        # Keep browser open briefly
        logger.info(f"Task {task_id}: Browser will remain open for 3 seconds...")
        time.sleep(3)
        
        # Prepare results
        result = {
            "status": "completed",
            "result": {
                "page_title": page_title,
                "url": driver.current_url,
                "database_file": db_path,
                "json_export_file": json_filename,
                "rows_extracted": len(rows_data),
                "structured_records": structured_count,
                "columns": len(headers),
                "headers": headers,
                "completed_at": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Task {task_id}: Automation completed successfully")
        logger.info(f"Task {task_id}: Created {structured_count} structured records in database")
        return result
        
    except Exception as e:
        logger.error(f"Task {task_id}: Error - {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Task {task_id}: Traceback - {error_trace}")
        
        # Save error to database
        save_task_info(task_id, url, "failed", error_message=str(e), db_path=db_path)
        
        return {
            "status": "failed",
            "error": str(e),
            "traceback": error_trace
        }
        
    finally:
        if driver:
            driver.quit()
            logger.info(f"Task {task_id}: Browser closed")