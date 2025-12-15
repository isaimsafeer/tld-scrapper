from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import Dict, Any, List, Optional
import logging
import uuid
from datetime import datetime
import os
from multiprocessing import Pool
import math
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

# Import automation function
from automation.automate import run_automation, init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL connection string
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql:"
)

# FastAPI app
app = FastAPI(
    title="Selenium Automation API",
    description="API to trigger web automation tasks and query pricing data",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@contextmanager
def get_db_connection():
    """Context manager for PostgreSQL database connections"""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise e
    finally:
        conn.close()


class AutomationRequest(BaseModel):
    url: HttpUrl
    timeout: int = Field(default=20, ge=5, le=300)
    headless: bool = Field(default=True)
    start_page: int = Field(default=1, ge=1, description="Starting page number")
    end_page: Optional[int] = Field(default=None, ge=1, description="Ending page number (None for all pages)")
    parallel: bool = Field(default=False, description="Enable parallel scraping with multiple workers")
    num_workers: Optional[int] = Field(default=8, ge=1, le=20, description="Number of parallel workers (only if parallel=True)")


class AutomationResponse(BaseModel):
    """Response model for automation"""
    task_id: str
    status: str
    message: str
    workers: Optional[int] = None
    estimated_time_minutes: Optional[float] = None


class PricingRecord(BaseModel):
    """Model for a single pricing record"""
    tld: str
    registrar: str
    operation: str
    price: float
    currency: str
    years: int
    pricing_notes: Optional[str]
    promo_code: Optional[str]
    scraped_at: str
    source_url: str
    column_name: str


class PricingResponse(BaseModel):
    """Response model for pricing data"""
    total_records: int
    filtered_records: int
    data: List[Dict[str, Any]]
    filters_applied: Dict[str, Any]


def init_postgres_database():
    """
    Initialize PostgreSQL database with required tables
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Create raw_data table
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

        # Create structured_pricing table
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

        # Create scraping_tasks table
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

        # Create indexes
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

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status 
            ON scraping_tasks(status)
        """)

        conn.commit()
        logger.info("PostgreSQL database initialized successfully")


# ============================================================
# PARALLEL SCRAPING FUNCTIONS
# ============================================================

def scrape_worker_process(args):
    """
    Worker function for parallel scraping (runs in separate process)
    
    Args:
        args: Tuple of (worker_id, url, start_page, end_page, timeout, headless, db_url, master_task_id)
    
    Returns:
        Dict with worker results
    """
    worker_id, url, start_page, end_page, timeout, headless, db_url, master_task_id = args
    
    # Create unique task ID for this worker
    task_id = f"{master_task_id}_worker_{worker_id}"
    
    logger.info(f"Worker {worker_id}: Starting pages {start_page}-{end_page}")
    
    try:
        result = run_automation(
            task_id=task_id,
            url=url,
            timeout=timeout,
            headless=headless,
            start_page=start_page,
            end_page=end_page,
            db_url=db_url
        )
        
        if result["status"] == "completed":
            rows = result["result"]["rows_extracted"]
            records = result["result"]["structured_records"]
            logger.info(f"Worker {worker_id}: Completed - {rows} rows, {records} records")
            return {
                "worker_id": worker_id,
                "task_id": task_id,
                "start_page": start_page,
                "end_page": end_page,
                "success": True,
                "rows_extracted": rows,
                "structured_records": records
            }
        else:
            logger.error(f"Worker {worker_id}: Failed")
            return {
                "worker_id": worker_id,
                "task_id": task_id,
                "start_page": start_page,
                "end_page": end_page,
                "success": False,
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        logger.exception(f"Worker {worker_id}: Exception occurred")
        return {
            "worker_id": worker_id,
            "task_id": task_id,
            "start_page": start_page,
            "end_page": end_page,
            "success": False,
            "error": str(e)
        }


def divide_pages_among_workers(total_pages: int, num_workers: int) -> List[tuple]:
    """Divide page ranges among workers evenly"""
    pages_per_worker = math.ceil(total_pages / num_workers)
    page_ranges = []
    
    for i in range(num_workers):
        start_page = i * pages_per_worker + 1
        end_page = min((i + 1) * pages_per_worker, total_pages)
        
        if start_page <= total_pages:
            page_ranges.append((start_page, end_page))
    
    return page_ranges


def execute_parallel_automation(master_task_id: str, url: str, total_pages: int, 
                                num_workers: int, timeout: int, headless: bool):
    """Execute parallel scraping with multiple workers"""
    start_time = datetime.now()
    
    logger.info(f"Master Task {master_task_id}: Starting parallel scraping")
    logger.info(f"Total pages: {total_pages}, Workers: {num_workers}")
    
    # Create master task record
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scraping_tasks 
                (task_id, url, status, started_at)
                VALUES (%s, %s, %s, %s)
            """, (master_task_id, url, "running", start_time))
    except Exception as e:
        logger.error(f"Failed to create master task: {e}")
    
    # Divide pages among workers
    page_ranges = divide_pages_among_workers(total_pages, num_workers)
    
    logger.info(f"Page distribution:")
    for i, (start, end) in enumerate(page_ranges):
        logger.info(f"  Worker {i+1}: Pages {start}-{end} ({end-start+1} pages)")
    
    # Prepare worker arguments
    worker_args = [
        (i+1, url, start, end, timeout, headless, DATABASE_URL, master_task_id)
        for i, (start, end) in enumerate(page_ranges)
    ]
    
    # Run workers in parallel
    try:
        with Pool(processes=num_workers) as pool:
            results = pool.map(scrape_worker_process, worker_args)
        
        # Analyze results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        successful_workers = sum(1 for r in results if r["success"])
        total_rows = sum(r.get("rows_extracted", 0) for r in results if r["success"])
        total_records = sum(r.get("structured_records", 0) for r in results if r["success"])
        
        # Update master task
        status = "completed" if successful_workers == len(results) else "partial"
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE scraping_tasks 
                SET status = %s,
                    rows_extracted = %s,
                    structured_records = %s,
                    completed_at = %s,
                    page_title = %s
                WHERE task_id = %s
            """, (
                status,
                total_rows,
                total_records,
                end_time,
                f"Parallel scraping: {successful_workers}/{num_workers} workers succeeded",
                master_task_id
            ))
        
        logger.info(f"Master Task {master_task_id}: Completed in {duration/60:.1f} minutes")
        logger.info(f"Successful workers: {successful_workers}/{num_workers}")
        logger.info(f"Total rows: {total_rows}, Total records: {total_records}")
        
    except Exception as e:
        logger.exception(f"Master Task {master_task_id}: Failed")
        
        # Update master task as failed
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE scraping_tasks 
                    SET status = %s,
                        error_message = %s,
                        completed_at = %s
                    WHERE task_id = %s
                """, ("failed", str(e), datetime.now(), master_task_id))
        except:
            pass


def execute_automation_task(task_id: str, url: str, timeout: int, headless: bool, 
                           start_page: int = 1, end_page: Optional[int] = None):
    """Execute single automation task (non-parallel)"""
    logger.info(f"Task {task_id}: Starting automation for pages {start_page}-{end_page or 'all'}")
    
    result = run_automation(
        task_id, 
        url, 
        timeout, 
        headless, 
        start_page=start_page, 
        end_page=end_page, 
        db_url=DATABASE_URL
    )
    
    logger.info(f"Task {task_id}: Completed with status: {result.get('status')}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_all_pricing_data() -> List[Dict[str, Any]]:
    """Load all pricing data from PostgreSQL database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT tld, registrar, operation, price, currency, years, 
                       pricing_notes, promo_code, scraped_at, source_url, column_name
                FROM structured_pricing
                ORDER BY scraped_at DESC
            """)
            
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            
            logger.info(f"Loaded {len(data)} pricing records from database")
            return data
    except Exception as e:
        logger.error(f"Error loading pricing data: {e}")
        return []


def filter_pricing_data(
    tld: Optional[str] = None,
    registrar: Optional[str] = None,
    operation: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    currency: Optional[str] = None,
    years: Optional[int] = None,
    has_promo: Optional[bool] = None,
    task_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Filter pricing data based on various criteria using SQL"""
    query = """
        SELECT tld, registrar, operation, price, currency, years, 
               pricing_notes, promo_code, scraped_at, source_url, column_name
        FROM structured_pricing
        WHERE 1=1
    """
    params = []
    
    if tld:
        tld_clean = tld.lower()
        if not tld_clean.startswith('.'):
            tld_clean = '.' + tld_clean
        query += " AND LOWER(tld) = %s"
        params.append(tld_clean)
    
    if registrar:
        query += " AND LOWER(registrar) LIKE %s"
        params.append(f"%{registrar.lower()}%")
    
    if operation:
        query += " AND LOWER(operation) = %s"
        params.append(operation.lower())
    
    if min_price is not None:
        query += " AND price >= %s"
        params.append(min_price)
    
    if max_price is not None:
        query += " AND price <= %s"
        params.append(max_price)
    
    if currency:
        query += " AND UPPER(currency) = %s"
        params.append(currency.upper())
    
    if years is not None:
        query += " AND years = %s"
        params.append(years)
    
    if has_promo is not None:
        if has_promo:
            query += " AND promo_code IS NOT NULL AND promo_code != ''"
        else:
            query += " AND (promo_code IS NULL OR promo_code = '')"
    
    if task_id:
        query += " AND task_id LIKE %s"
        params.append(f"{task_id}%")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error filtering pricing data: {e}")
        return []


def sort_pricing_data(
    data: List[Dict[str, Any]],
    sort_by: str = 'price',
    sort_order: str = 'asc'
) -> List[Dict[str, Any]]:
    """Sort pricing data"""
    reverse = sort_order.lower() == 'desc'
    
    if sort_by == 'price':
        return sorted(data, key=lambda x: float(x.get('price', 0)), reverse=reverse)
    elif sort_by == 'tld':
        return sorted(data, key=lambda x: x.get('tld', ''), reverse=reverse)
    elif sort_by == 'registrar':
        return sorted(data, key=lambda x: x.get('registrar', ''), reverse=reverse)
    elif sort_by == 'scraped_at':
        return sorted(data, key=lambda x: str(x.get('scraped_at', '')), reverse=reverse)
    else:
        return data


# ============================================================
# API ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("Initializing PostgreSQL database...")
    init_postgres_database()
    logger.info("Database initialized successfully")


@app.post("/automate", response_model=AutomationResponse)
async def trigger_automation(
    request: AutomationRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger web automation in the background
    
    Supports both single-threaded and parallel scraping modes.
    """
    task_id = str(uuid.uuid4())
    
    if request.parallel and request.end_page:
        total_pages = request.end_page
        num_workers = request.num_workers or 8
        estimated_time = total_pages / (num_workers * 2)
        
        logger.info(f"Task {task_id}: Parallel mode with {num_workers} workers")
        logger.info(f"Estimated completion: {estimated_time:.1f} minutes")
        
        background_tasks.add_task(
            execute_parallel_automation,
            task_id,
            str(request.url),
            total_pages,
            num_workers,
            request.timeout,
            request.headless
        )
        
        return AutomationResponse(
            task_id=task_id,
            status="queued",
            message=f"Parallel automation with {num_workers} workers queued successfully",
            workers=num_workers,
            estimated_time_minutes=round(estimated_time, 1)
        )
    else:
        logger.info(f"Task {task_id}: Single-threaded mode")
        
        background_tasks.add_task(
            execute_automation_task,
            task_id,
            str(request.url),
            request.timeout,
            request.headless,
            request.start_page,
            request.end_page
        )
        
        return AutomationResponse(
            task_id=task_id,
            status="queued",
            message="Automation task queued successfully"
        )


@app.get("/automation/{task_id}")
async def get_automation_status(task_id: str):
    """Get the status of an automation task"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT task_id, url, status, page_title, rows_extracted, 
                       structured_records, columns, headers, started_at, 
                       completed_at, error_message
                FROM scraping_tasks
                WHERE task_id = %s
            """, (task_id,))
            
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Task not found")
            
            result = dict(row)
            
            # Check for worker tasks (parallel mode)
            cursor.execute("""
                SELECT task_id, status, rows_extracted, structured_records, 
                       started_at, completed_at
                FROM scraping_tasks
                WHERE task_id LIKE %s
                ORDER BY task_id
            """, (f"{task_id}_worker_%",))
            
            worker_tasks = [dict(r) for r in cursor.fetchall()]
            
            if worker_tasks:
                result['parallel'] = True
                result['workers'] = worker_tasks
                result['total_workers'] = len(worker_tasks)
                result['completed_workers'] = sum(1 for w in worker_tasks if w['status'] == 'completed')
            
            # Convert datetime to ISO format
            for key in ['started_at', 'completed_at', 'scraped_at']:
                if key in result and result[key]:
                    result[key] = result[key].isoformat()
            
            return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/automations")
async def list_automation_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all automation tasks (excluding worker sub-tasks)"""
    query = """
        SELECT task_id, url, status, page_title, rows_extracted, 
               structured_records, started_at, completed_at
        FROM scraping_tasks
        WHERE task_id NOT LIKE %s
    """
    params = ['%_worker_%']
    
    if status:
        query += " AND status = %s"
        params.append(status)
    
    query += " ORDER BY started_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            tasks = [dict(row) for row in rows]
            
            # Convert datetime to ISO format
            for task in tasks:
                for key in ['started_at', 'completed_at']:
                    if key in task and task[key]:
                        task[key] = task[key].isoformat()
            
            # Get total count
            count_query = "SELECT COUNT(*) as total FROM scraping_tasks WHERE task_id NOT LIKE %s"
            count_params = ['%_worker_%']
            if status:
                count_query += " AND status = %s"
                count_params.append(status)
            
            cursor.execute(count_query, count_params)
            total = cursor.fetchone()['total']
            
            return JSONResponse(content={
                "total": total,
                "limit": limit,
                "offset": offset,
                "tasks": tasks
            })
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pricing", response_model=PricingResponse)
async def get_pricing_data(
    tld: Optional[str] = Query(None),
    registrar: Optional[str] = Query(None),
    operation: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    currency: Optional[str] = Query(None),
    years: Optional[int] = Query(None, ge=1),
    has_promo: Optional[bool] = Query(None),
    task_id: Optional[str] = Query(None),
    sort_by: str = Query('price'),
    sort_order: str = Query('asc'),
    limit: Optional[int] = Query(None, ge=1),
    offset: int = Query(0, ge=0)
):
    """Get pricing data with filters and sorting"""
    all_data = load_all_pricing_data()
    
    filtered_data = filter_pricing_data(
        tld=tld, registrar=registrar, operation=operation,
        min_price=min_price, max_price=max_price, currency=currency,
        years=years, has_promo=has_promo, task_id=task_id
    )
    
    sorted_data = sort_pricing_data(filtered_data, sort_by=sort_by, sort_order=sort_order)
    
    paginated_data = sorted_data[offset:]
    if limit:
        paginated_data = paginated_data[:limit]
    
    # Convert datetime and decimal to serializable types
    for record in paginated_data:
        if 'scraped_at' in record and record['scraped_at']:
            record['scraped_at'] = record['scraped_at'].isoformat() if hasattr(record['scraped_at'], 'isoformat') else str(record['scraped_at'])
        if 'price' in record:
            record['price'] = float(record['price'])
    
    filters_applied = {k: v for k, v in {
        "tld": tld, "registrar": registrar, "operation": operation,
        "min_price": min_price, "max_price": max_price, "currency": currency,
        "years": years, "has_promo": has_promo, "task_id": task_id,
        "sort_by": sort_by, "sort_order": sort_order, "limit": limit, "offset": offset
    }.items() if v is not None}
    
    return PricingResponse(
        total_records=len(all_data),
        filtered_records=len(sorted_data),
        data=paginated_data,
        filters_applied=filters_applied
    )


@app.get("/pricing/cheapest")
async def get_cheapest_pricing(
    tld: Optional[str] = Query(None),
    operation: Optional[str] = Query(None),
    currency: Optional[str] = Query("USD"),
    group_by: str = Query("tld")
):
    """Get the cheapest pricing for each TLD/operation combination"""
    query = """
        SELECT tld, registrar, operation, MIN(price) as price, currency, 
               years, pricing_notes, promo_code, scraped_at, source_url, column_name
        FROM structured_pricing
        WHERE 1=1
    """
    params = []
    
    if tld:
        tld_clean = tld.lower()
        if not tld_clean.startswith('.'):
            tld_clean = '.' + tld_clean
        query += " AND LOWER(tld) = %s"
        params.append(tld_clean)
    
    if operation:
        query += " AND LOWER(operation) = %s"
        params.append(operation.lower())
    
    if currency:
        query += " AND UPPER(currency) = %s"
        params.append(currency.upper())
    
    if group_by == "tld":
        query += " GROUP BY tld, operation, registrar, currency, years, pricing_notes, promo_code, scraped_at, source_url, column_name"
    elif group_by == "operation":
        query += " GROUP BY operation, tld, registrar, currency, years, pricing_notes, promo_code, scraped_at, source_url, column_name"
    elif group_by == "registrar":
        query += " GROUP BY registrar, tld, operation, currency, years, pricing_notes, promo_code, scraped_at, source_url, column_name"
    
    query += " ORDER BY price ASC"
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            
            # Convert types
            for record in result:
                if 'price' in record:
                    record['price'] = float(record['price'])
                if 'scraped_at' in record and record['scraped_at']:
                    record['scraped_at'] = record['scraped_at'].isoformat()
            
            return JSONResponse(content={
                "total_results": len(result),
                "group_by": group_by,
                "data": result
            })
    except Exception as e:
        logger.error(f"Error getting cheapest pricing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pricing/stats")
async def get_pricing_stats():
    """Get statistics about the pricing data"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT COUNT(*) as total FROM structured_pricing")
            total_records = cursor.fetchone()['total']
            
            cursor.execute("SELECT DISTINCT tld FROM structured_pricing ORDER BY tld")
            tlds = [row['tld'] for row in cursor.fetchall()]
            
            cursor.execute("SELECT DISTINCT registrar FROM structured_pricing ORDER BY registrar")
            registrars = [row['registrar'] for row in cursor.fetchall()]
            
            cursor.execute("SELECT DISTINCT operation FROM structured_pricing ORDER BY operation")
            operations = [row['operation'] for row in cursor.fetchall()]
            
            cursor.execute("SELECT DISTINCT currency FROM structured_pricing ORDER BY currency")
            currencies = [row['currency'] for row in cursor.fetchall()]
            
            cursor.execute("""
                SELECT 
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(price) as avg_price,
                    COUNT(price) as price_count
                FROM structured_pricing
                WHERE price IS NOT NULL
            """)
            price_stats = dict(cursor.fetchone())
            
            # Convert Decimal to float
            for key in ['min_price', 'max_price', 'avg_price']:
                if price_stats[key] is not None:
                    price_stats[key] = float(price_stats[key])
            
            cursor.execute("""
                SELECT COUNT(*) as promo_count
                FROM structured_pricing
                WHERE promo_code IS NOT NULL AND promo_code != ''
            """)
            promo_count = cursor.fetchone()['promo_count']
            
            stats = {
                "total_records": total_records,
                "unique_tlds": len(tlds),
                "tlds": tlds,
                "unique_registrars": len(registrars),
                "registrars": registrars,
                "operations": operations,
                "currencies": currencies,
                "price_stats": {
                    "min": price_stats['min_price'] or 0,
                    "max": price_stats['max_price'] or 0,
                    "avg": price_stats['avg_price'] or 0,
                    "count": price_stats['price_count'] or 0
                },
                "records_with_promo": promo_count
            }
            
            return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting pricing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pricing/compare")
async def compare_registrars(
    tld: str = Query(..., description="TLD to compare"),
    operation: str = Query("register", description="Operation to compare")
):
    """Compare prices across all registrars for a specific TLD and operation"""
    tld_clean = tld.lower()
    if not tld_clean.startswith('.'):
        tld_clean = '.' + tld_clean
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT registrar, MIN(price) as price, currency, 
                       pricing_notes, promo_code, source_url
                FROM structured_pricing
                WHERE LOWER(tld) = %s AND LOWER(operation) = %s
                GROUP BY registrar, currency, pricing_notes, promo_code, source_url
                ORDER BY price ASC
            """, (tld_clean, operation.lower()))
            
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            
            # Convert Decimal to float
            for record in result:
                if 'price' in record:
                    record['price'] = float(record['price'])
            
            if not result:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No pricing data found for {tld} ({operation})"
                )
            
            return JSONResponse(content={
                "tld": tld_clean,
                "operation": operation,
                "registrar_count": len(result),
                "cheapest": result[0] if result else None,
                "most_expensive": result[-1] if result else None,
                "all_prices": result
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing registrars: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/pricing")
async def delete_pricing_data(
    task_id: Optional[str] = Query(None),
    confirm: bool = Query(False)
):
    """Delete pricing data (use with caution!)"""
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Deletion requires confirm=true parameter"
        )
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if task_id:
                cursor.execute("DELETE FROM structured_pricing WHERE task_id LIKE %s", (f"{task_id}%",))
                cursor.execute("DELETE FROM raw_data WHERE task_id LIKE %s", (f"{task_id}%",))
                cursor.execute("DELETE FROM scraping_tasks WHERE task_id LIKE %s", (f"{task_id}%",))
                deleted_count = cursor.rowcount
                message = f"Deleted data for task {task_id}"
            else:
                cursor.execute("DELETE FROM structured_pricing")
                cursor.execute("DELETE FROM raw_data")
                cursor.execute("DELETE FROM scraping_tasks")
                deleted_count = cursor.rowcount
                message = "Deleted all pricing data"
            
            return JSONResponse(content={
                "message": message,
                "deleted_records": deleted_count
            })
    except Exception as e:
        logger.error(f"Error deleting pricing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)