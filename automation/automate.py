#!/usr/bin/env python3
"""
Full automation script with PostgreSQL support for parallel scraping.
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import time
import re
import psycopg2
from psycopg2.extras import execute_values
from contextlib import contextmanager
import os
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@contextmanager
def get_db_connection(db_url: str):
    """Context manager for PostgreSQL database connections"""
    conn = psycopg2.connect(db_url)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database(db_url: str):
    """Initialize PostgreSQL database with required tables"""
    with get_db_connection(db_url) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_data (
                id SERIAL PRIMARY KEY,
                task_id TEXT NOT NULL,
                row_number INTEGER,
                domain TEXT,
                column_name TEXT,
                cell_value TEXT,
                scraped_at TIMESTAMP,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS structured_pricing (
                id SERIAL PRIMARY KEY,
                task_id TEXT NOT NULL,
                tld TEXT NOT NULL,
                registrar TEXT NOT NULL,
                operation TEXT NOT NULL,
                price DECIMAL(10, 2),
                currency TEXT DEFAULT 'USD',
                years INTEGER DEFAULT 1,
                pricing_notes TEXT,
                promo_code TEXT,
                scraped_at TIMESTAMP,
                source_url TEXT,
                column_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(task_id, tld, registrar, operation, column_name)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scraping_tasks (
                task_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                status TEXT NOT NULL,
                page_title TEXT,
                rows_extracted INTEGER DEFAULT 0,
                structured_records INTEGER DEFAULT 0,
                columns INTEGER DEFAULT 0,
                headers JSONB,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT
            )
        """)

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


def setup_driver(headless: bool = True):
    """Setup and return a Chrome webdriver instance"""
    chrome_options = Options()

    if headless:
        chrome_options.add_argument("--headless=new")

    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-setuid-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-background-networking")
    chrome_options.add_argument("--disable-default-apps")
    chrome_options.add_argument("--disable-sync")
    chrome_options.add_argument("--disable-translate")
    chrome_options.add_argument("--hide-scrollbars")
    chrome_options.add_argument("--metrics-recording-only")
    chrome_options.add_argument("--mute-audio")
    chrome_options.add_argument("--no-first-run")
    chrome_options.add_argument("--safebrowsing-disable-auto-update")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    chromedriver_path = shutil.which("chromedriver")
    
    try:
        if chromedriver_path:
            logger.info(f"Using system chromedriver at: {chromedriver_path}")
            service = Service(chromedriver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            logger.info("No chromedriver path specified, letting Selenium auto-detect")
            driver = webdriver.Chrome(options=chrome_options)
        
        logger.info("Chrome webdriver started successfully")
        return driver
    except Exception as e:
        logger.exception("Failed to start Chrome webdriver")
        raise RuntimeError(
            f"Failed to start Chrome webdriver. "
            f"Make sure Chrome and chromedriver are installed on your system. "
            f"Chrome path: {shutil.which('google-chrome') or 'not found'}\n"
            f"Chromedriver path: {chromedriver_path or 'not found'}\n"
            f"Original error: {e}"
        ) from e


def extract_registrar_from_url(url: str) -> str:
    """Extract registrar name from URL"""
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if match:
        domain = match.group(1)
        parts = domain.split('.')
        if len(parts) >= 2:
            return parts[0]
        return domain
    return "unknown"


def parse_price_data(cell_value: str) -> Dict[str, Any]:
    """Parse price information from cell value"""
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

    if lines:
        result["registrar"] = lines[0].strip().lower()
        logger.debug(f"Extracted registrar: {result['registrar']}")

    promo_match = re.search(r'Promo code[:\s]*([A-Za-z0-9_-]+)', cell_value, re.IGNORECASE)
    if promo_match:
        result["promo_code"] = promo_match.group(1)
        result["pricing_notes"] = f"promo code: {promo_match.group(1)}"
        logger.debug(f"Found promo code: {result['promo_code']}")

    price_patterns = [
        (r'\$(\d+\.?\d*)', 'USD'),
        (r'£\s*(\d+\.?\d*)', 'GBP'),
        (r'€\s*(\d+\.?\d*)', 'EUR'),
    ]

    for pattern, currency in price_patterns:
        price_match = re.search(pattern, cell_value)
        if price_match:
            try:
                result["price"] = float(price_match.group(1))
                result["currency"] = currency
                logger.debug(f"Found price: {result['price']} {currency}")
                break
            except ValueError:
                logger.warning(f"Could not convert extracted price to float: {price_match.group(1)}")

    if result["price"] is None:
        logger.warning(f"Could not extract price from: {repr(cell_value[:100])}")

    low = cell_value.lower()
    if "registration" in low:
        if result["pricing_notes"]:
            result["pricing_notes"] += ", registration price"
        else:
            result["pricing_notes"] = "registration price"

    if "renewal" in low and "registration" in low:
        if result["pricing_notes"]:
            result["pricing_notes"] += ", includes renewal"
        else:
            result["pricing_notes"] = "includes renewal"

    return result


def map_column_to_operation(column_name: str) -> str:
    """Map column header to operation type"""
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
                        task_id: str, url: str, db_url: str):
    """Save raw scraped data to database"""
    scraped_at = datetime.now()

    with get_db_connection(db_url) as conn:
        cursor = conn.cursor()
        
        # Batch insert for better performance
        insert_data = []
        for row_num, row in enumerate(rows_data, start=1):
            domain = row[0] if row else ""
            
            for col_idx, cell_value in enumerate(row[1:], start=1):
                if col_idx < len(headers):
                    column_name = headers[col_idx]
                    insert_data.append((
                        task_id, row_num, domain, column_name, 
                        cell_value, scraped_at, url
                    ))
        
        if insert_data:
            execute_values(cursor, """
                INSERT INTO raw_data (task_id, row_number, domain, column_name, 
                                    cell_value, scraped_at, source_url)
                VALUES %s
            """, insert_data)

        logger.info(f"Task {task_id}: Saved {len(rows_data)} raw data rows to database")


def process_and_save_structured_data(headers: List[str], rows_data: List[List[str]],
                                     task_id: str, url: str, db_url: str) -> int:
    """Process raw data and save structured pricing data to database"""
    scraped_at = datetime.now()
    registrar_from_url = extract_registrar_from_url(url)
    records_count = 0

    with get_db_connection(db_url) as conn:
        cursor = conn.cursor()

        for row_idx, row in enumerate(rows_data, start=2):
            if not row or not any(row):
                continue

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

                try:
                    cursor.execute("""
                        INSERT INTO structured_pricing 
                        (task_id, tld, registrar, operation, price, currency, years, 
                         pricing_notes, promo_code, scraped_at, source_url, column_name)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (task_id, tld, registrar, operation, column_name) 
                        DO UPDATE SET price = EXCLUDED.price
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
                except Exception as e:
                    logger.warning(f"Task {task_id}: Error inserting record - {e}")

        logger.info(f"Task {task_id}: Saved {records_count} structured records to database")

    return records_count


def save_task_info(task_id: str, url: str, status: str,
                   page_title: str = None, rows_extracted: int = 0,
                   structured_records: int = 0, columns: int = 0,
                   headers: List[str] = None, error_message: str = None,
                   db_url: str = None):
    """Save or update task information in database"""
    with get_db_connection(db_url) as conn:
        cursor = conn.cursor()

        now = datetime.now()

        # Check if task exists
        cursor.execute("SELECT task_id, started_at FROM scraping_tasks WHERE task_id = %s", (task_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing task
            cursor.execute("""
                UPDATE scraping_tasks 
                SET status = %s,
                    page_title = %s,
                    rows_extracted = %s,
                    structured_records = %s,
                    columns = %s,
                    headers = %s,
                    completed_at = %s,
                    error_message = %s
                WHERE task_id = %s
            """, (status, page_title, rows_extracted, structured_records,
                  columns, psycopg2.extras.Json(headers) if headers else None,
                  now if status in ['completed', 'failed'] else None,
                  error_message, task_id))
        else:
            # Insert new task
            cursor.execute("""
                INSERT INTO scraping_tasks 
                (task_id, url, status, page_title, rows_extracted, structured_records, 
                 columns, headers, started_at, completed_at, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (task_id, url, status, page_title, rows_extracted, structured_records,
                  columns, psycopg2.extras.Json(headers) if headers else None,
                  now, now if status in ['completed', 'failed'] else None, error_message))


def run_automation(task_id: str, url: str, timeout: int, headless: bool,
                   start_page: int = 1, end_page: Optional[int] = None,
                   db_url: str = None) -> Dict[str, Any]:
    """Main automation function"""
    driver = None

    # Initialize database
    init_database(db_url)

    try:
        logger.info(f"Task {task_id}: Starting automation for {url}")
        logger.info(f"Task {task_id}: Page range - Start: {start_page}, End: {end_page or 'all'}")

        save_task_info(task_id, url, "started", db_url=db_url)

        driver = setup_driver(headless=headless)
        driver.get(url)
        logger.info(f"Task {task_id}: Navigated to {url}")

        wait = WebDriverWait(driver, timeout)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        page_title = driver.title
        logger.info(f"Task {task_id}: Page title: {page_title}")

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
        try:
            last_page_element = driver.find_element(By.XPATH,
                "//div[@id='DataTables_Table_0_paginate']//li[last()-1]")
            total_pages = int(last_page_element.text)
        except Exception:
            try:
                page_items = driver.find_elements(By.XPATH, "//div[@id='DataTables_Table_0_paginate']//li")
                total_pages = max(1, len(page_items) - 2)
            except Exception:
                total_pages = 1

        if end_page is None:
            pages_to_scrape = total_pages
        else:
            pages_to_scrape = min(end_page, total_pages)
        
        if start_page > total_pages:
            logger.warning(f"Task {task_id}: start_page ({start_page}) exceeds total pages ({total_pages})")
            start_page = total_pages
        
        if start_page < 1:
            logger.warning(f"Task {task_id}: start_page ({start_page}) is less than 1, setting to 1")
            start_page = 1
        
        actual_start = start_page
        actual_end = pages_to_scrape
        
        logger.info(f"Task {task_id}: Total pages available: {total_pages}")
        logger.info(f"Task {task_id}: Scraping pages {actual_start} to {actual_end}")

        # Navigate to start_page if not page 1
        if start_page > 1:
            logger.info(f"Task {task_id}: Navigating to start page {start_page}...")
            for nav_page in range(1, start_page):
                try:
                    next_button = driver.find_element(By.XPATH,
                        "//*[@id='DataTables_Table_0_next']/a")
                    
                    parent_li = driver.find_element(By.XPATH,
                        "//*[@id='DataTables_Table_0_next']")
                    if "disabled" in parent_li.get_attribute("class"):
                        logger.warning(f"Task {task_id}: Cannot navigate further, next button disabled")
                        break
                    
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(1)
                    next_button.click()
                    logger.info(f"Task {task_id}: Navigated to page {nav_page + 1}")
                    time.sleep(2)
                except Exception as nav_error:
                    logger.error(f"Task {task_id}: Could not navigate to page {nav_page + 1}: {nav_error}")
                    raise

        # Scrape pages
        for page_num in range(actual_start, actual_end + 1):
            logger.info(f"Task {task_id}: Extracting data from page {page_num}...")
            time.sleep(2)

            table_rows = driver.find_elements(By.XPATH,
                "(//table[@id='DataTables_Table_0'])[1]//tbody//tr")
            logger.info(f"Task {task_id}: Found {len(table_rows)} rows on page {page_num}")

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

            if page_num < actual_end:
                try:
                    next_button = driver.find_element(By.XPATH,
                        "//*[@id='DataTables_Table_0_next']/a")

                    parent_li = driver.find_element(By.XPATH,
                        "//*[@id='DataTables_Table_0_next']")
                    if "disabled" in parent_li.get_attribute("class"):
                        logger.info(f"Task {task_id}: Next button disabled, stopping pagination")
                        break

                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(1)
                    next_button.click()
                    logger.info(f"Task {task_id}: Clicked Next to page {page_num + 1}")
                    time.sleep(2)
                except Exception as click_error:
                    logger.warning(f"Task {task_id}: Could not click Next: {click_error}")
                    break

        logger.info(f"Task {task_id}: Extracted {len(rows_data)} total rows from pages {actual_start}-{page_num}")

        # Save to database
        save_raw_data_to_db(headers, rows_data, task_id, url, db_url)
        structured_count = process_and_save_structured_data(
            headers, rows_data, task_id, url, db_url)

        # Update task info
        save_task_info(
            task_id, url, "completed", page_title,
            len(rows_data), structured_count, len(headers), headers,
            db_url=db_url
        )

        logger.info(f"Task {task_id}: Browser will remain open for 3 seconds...")
        time.sleep(3)

        result = {
            "status": "completed",
            "result": {
                "page_title": page_title,
                "url": driver.current_url,
                "rows_extracted": len(rows_data),
                "structured_records": structured_count,
                "columns": len(headers),
                "headers": headers,
                "pages_scraped": f"{actual_start}-{page_num}",
                "total_pages_available": total_pages,
                "completed_at": datetime.now().isoformat()
            }
        }

        logger.info(f"Task {task_id}: Automation completed successfully")
        return result

    except Exception as e:
        logger.error(f"Task {task_id}: Error - {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Task {task_id}: Traceback - {error_trace}")

        try:
            save_task_info(task_id, url, "failed", error_message=str(e), db_url=db_url)
        except Exception as save_err:
            logger.exception(f"Failed to save task error info: {save_err}")

        return {
            "status": "failed",
            "error": str(e),
            "traceback": error_trace
        }

    finally:
        if driver:
            try:
                driver.quit()
                logger.info(f"Task {task_id}: Browser closed")
            except Exception:
                logger.exception("Error while closing the browser")


if __name__ == "__main__":
    import uuid
    test_task_id = str(uuid.uuid4())
    test_url = "https://tld-list.com/"
    test_db_url = os.environ.get("DATABASE_URL", "postgresql")
    
    try:
        res = run_automation(test_task_id, test_url, timeout=20, headless=True, 
                            start_page=1, end_page=3, db_url=test_db_url)
        print(res)
    except Exception as ex:
        logger.exception(f"Top-level error: {ex}")