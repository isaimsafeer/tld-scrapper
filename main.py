from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List, Optional
import logging
import uuid
from datetime import datetime
import os
import sqlite3
from contextlib import contextmanager
from fastapi.staticfiles import StaticFiles

# Import automation function
from automation.automate import run_automation, init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "domain_pricing.db"

# FastAPI app
app = FastAPI(
    title="Selenium Automation API",
    description="API to trigger web automation tasks and query pricing data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@contextmanager
def get_db_connection(db_path: str = DB_PATH):
    """Context manager for database connections"""
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


class AutomationRequest(BaseModel):
    """Request model for automation"""
    url: HttpUrl = "https://www.example.com"
    headless: bool = True
    timeout: int = 10


class AutomationResponse(BaseModel):
    """Response model for automation"""
    task_id: str
    status: str
    message: str


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


def load_all_pricing_data() -> List[Dict[str, Any]]:
    """
    Load all pricing data from SQLite database
    
    Returns:
        List of all pricing records
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
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
    """
    Filter pricing data based on various criteria using SQL
    
    Args:
        tld: Filter by TLD (e.g., '.com', '.io')
        registrar: Filter by registrar name
        operation: Filter by operation type (register, renew, transfer, multi_year)
        min_price: Minimum price
        max_price: Maximum price
        currency: Filter by currency (USD, GBP, EUR)
        years: Filter by number of years
        has_promo: Filter records with/without promo codes
        task_id: Filter by specific scraping task
        
    Returns:
        Filtered list of pricing records
    """
    query = """
        SELECT tld, registrar, operation, price, currency, years, 
               pricing_notes, promo_code, scraped_at, source_url, column_name
        FROM structured_pricing
        WHERE 1=1
    """
    params = []
    
    # Build dynamic WHERE clause
    if tld:
        tld_clean = tld.lower()
        if not tld_clean.startswith('.'):
            tld_clean = '.' + tld_clean
        query += " AND LOWER(tld) = ?"
        params.append(tld_clean)
    
    if registrar:
        query += " AND LOWER(registrar) LIKE ?"
        params.append(f"%{registrar.lower()}%")
    
    if operation:
        query += " AND LOWER(operation) = ?"
        params.append(operation.lower())
    
    if min_price is not None:
        query += " AND price >= ?"
        params.append(min_price)
    
    if max_price is not None:
        query += " AND price <= ?"
        params.append(max_price)
    
    if currency:
        query += " AND UPPER(currency) = ?"
        params.append(currency.upper())
    
    if years is not None:
        query += " AND years = ?"
        params.append(years)
    
    if has_promo is not None:
        if has_promo:
            query += " AND promo_code IS NOT NULL AND promo_code != ''"
        else:
            query += " AND (promo_code IS NULL OR promo_code = '')"
    
    if task_id:
        query += " AND task_id = ?"
        params.append(task_id)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
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
    """
    Sort pricing data
    
    Args:
        data: List of pricing records
        sort_by: Field to sort by (price, tld, registrar, scraped_at)
        sort_order: Sort order (asc or desc)
        
    Returns:
        Sorted list of pricing records
    """
    reverse = sort_order.lower() == 'desc'
    
    if sort_by == 'price':
        return sorted(data, key=lambda x: x.get('price', 0), reverse=reverse)
    elif sort_by == 'tld':
        return sorted(data, key=lambda x: x.get('tld', ''), reverse=reverse)
    elif sort_by == 'registrar':
        return sorted(data, key=lambda x: x.get('registrar', ''), reverse=reverse)
    elif sort_by == 'scraped_at':
        return sorted(data, key=lambda x: x.get('scraped_at', ''), reverse=reverse)
    else:
        return data


def execute_automation_task(task_id: str, url: str, timeout: int, headless: bool):
    """
    Wrapper function to execute automation
    
    Args:
        task_id: Unique task identifier
        url: Target website URL
        timeout: Maximum wait time for elements
        headless: Run in headless mode
    """
    logger.info(f"Starting automation task {task_id}")
    
    # Run automation (now saves to database)
    result = run_automation(task_id, url, timeout, headless, db_path=DB_PATH)
    
    logger.info(f"Automation task {task_id} completed with status: {result.get('status')}")


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("Initializing database...")
    init_database(DB_PATH)
    logger.info("Database initialized successfully")


@app.post("/automate", response_model=AutomationResponse)
async def trigger_automation(
    request: AutomationRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger web automation in the background
    
    Args:
        request: Automation configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Task ID and status
    """
    task_id = str(uuid.uuid4())
    
    logger.info(f"Task {task_id} queued for {request.url}")
    
    # Add automation to background tasks
    background_tasks.add_task(
        execute_automation_task,
        task_id,
        str(request.url),
        request.timeout,
        request.headless
    )
    
    return AutomationResponse(
        task_id=task_id,
        status="queued",
        message="Automation task queued successfully"
    )


@app.get("/automation/{task_id}")
async def get_automation_status(task_id: str):
    """
    Get the status of an automation task
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task status and results
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT task_id, url, status, page_title, rows_extracted, 
                       structured_records, columns, headers, started_at, 
                       completed_at, error_message
                FROM scraping_tasks
                WHERE task_id = ?
            """, (task_id,))
            
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Task not found")
            
            result = dict(row)
            
            # Parse headers JSON if present
            if result.get('headers'):
                import json
                try:
                    result['headers'] = json.loads(result['headers'])
                except:
                    pass
            
            return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/automations")
async def list_automation_tasks(
    status: Optional[str] = Query(None, description="Filter by status (started, completed, failed)"),
    limit: int = Query(50, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Offset for pagination", ge=0)
):
    """
    List all automation tasks
    
    Query Parameters:
        - status: Filter by task status
        - limit: Maximum number of results
        - offset: Offset for pagination
    
    Returns:
        List of automation tasks
    """
    query = """
        SELECT task_id, url, status, page_title, rows_extracted, 
               structured_records, started_at, completed_at
        FROM scraping_tasks
        WHERE 1=1
    """
    params = []
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            tasks = [dict(row) for row in rows]
            
            # Get total count
            count_query = "SELECT COUNT(*) as total FROM scraping_tasks WHERE 1=1"
            count_params = []
            if status:
                count_query += " AND status = ?"
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
    tld: Optional[str] = Query(None, description="Filter by TLD (e.g., 'com', '.com', 'io')"),
    registrar: Optional[str] = Query(None, description="Filter by registrar name"),
    operation: Optional[str] = Query(None, description="Filter by operation (register, renew, transfer, multi_year)"),
    min_price: Optional[float] = Query(None, description="Minimum price", ge=0),
    max_price: Optional[float] = Query(None, description="Maximum price", ge=0),
    currency: Optional[str] = Query(None, description="Filter by currency (USD, GBP, EUR)"),
    years: Optional[int] = Query(None, description="Filter by number of years", ge=1),
    has_promo: Optional[bool] = Query(None, description="Filter by promo code availability"),
    task_id: Optional[str] = Query(None, description="Filter by scraping task ID"),
    sort_by: str = Query('price', description="Sort by field (price, tld, registrar, scraped_at)"),
    sort_order: str = Query('asc', description="Sort order (asc, desc)"),
    limit: Optional[int] = Query(None, description="Limit number of results", ge=1),
    offset: int = Query(0, description="Offset for pagination", ge=0)
):
    """
    Get pricing data with filters and sorting
    
    Returns:
        Filtered and sorted pricing data
    """
    # Get total count
    all_data = load_all_pricing_data()
    
    # Apply filters
    filtered_data = filter_pricing_data(
        tld=tld,
        registrar=registrar,
        operation=operation,
        min_price=min_price,
        max_price=max_price,
        currency=currency,
        years=years,
        has_promo=has_promo,
        task_id=task_id
    )
    
    # Sort data
    sorted_data = sort_pricing_data(filtered_data, sort_by=sort_by, sort_order=sort_order)
    
    # Apply pagination
    paginated_data = sorted_data[offset:]
    if limit:
        paginated_data = paginated_data[:limit]
    
    # Track which filters were applied
    filters_applied = {
        "tld": tld,
        "registrar": registrar,
        "operation": operation,
        "min_price": min_price,
        "max_price": max_price,
        "currency": currency,
        "years": years,
        "has_promo": has_promo,
        "task_id": task_id,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "limit": limit,
        "offset": offset
    }
    
    # Remove None values
    filters_applied = {k: v for k, v in filters_applied.items() if v is not None}
    
    return PricingResponse(
        total_records=len(all_data),
        filtered_records=len(sorted_data),
        data=paginated_data,
        filters_applied=filters_applied
    )


@app.get("/pricing/cheapest")
async def get_cheapest_pricing(
    tld: Optional[str] = Query(None, description="Filter by TLD"),
    operation: Optional[str] = Query(None, description="Filter by operation (register, renew, transfer)"),
    currency: Optional[str] = Query("USD", description="Currency (USD, GBP, EUR)"),
    group_by: str = Query("tld", description="Group by (tld, operation, registrar)")
):
    """
    Get the cheapest pricing for each TLD/operation combination
    
    Returns:
        Cheapest pricing options
    """
    # Build SQL query for cheapest prices
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
        query += " AND LOWER(tld) = ?"
        params.append(tld_clean)
    
    if operation:
        query += " AND LOWER(operation) = ?"
        params.append(operation.lower())
    
    if currency:
        query += " AND UPPER(currency) = ?"
        params.append(currency.upper())
    
    # Group by clause based on group_by parameter
    if group_by == "tld":
        query += " GROUP BY tld, operation"
    elif group_by == "operation":
        query += " GROUP BY operation, tld"
    elif group_by == "registrar":
        query += " GROUP BY registrar, tld"
    else:
        query += " GROUP BY tld"
    
    query += " ORDER BY price ASC"
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            
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
    """
    Get statistics about the pricing data
    
    Returns:
        Statistics including available TLDs, registrars, price ranges, etc.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get basic counts
            cursor.execute("SELECT COUNT(*) as total FROM structured_pricing")
            total_records = cursor.fetchone()['total']
            
            # Get unique TLDs
            cursor.execute("SELECT DISTINCT tld FROM structured_pricing ORDER BY tld")
            tlds = [row['tld'] for row in cursor.fetchall()]
            
            # Get unique registrars
            cursor.execute("SELECT DISTINCT registrar FROM structured_pricing ORDER BY registrar")
            registrars = [row['registrar'] for row in cursor.fetchall()]
            
            # Get unique operations
            cursor.execute("SELECT DISTINCT operation FROM structured_pricing ORDER BY operation")
            operations = [row['operation'] for row in cursor.fetchall()]
            
            # Get unique currencies
            cursor.execute("SELECT DISTINCT currency FROM structured_pricing ORDER BY currency")
            currencies = [row['currency'] for row in cursor.fetchall()]
            
            # Get price statistics
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
            
            # Get promo count
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
    tld: str = Query(..., description="TLD to compare (e.g., 'com', '.com')"),
    operation: str = Query("register", description="Operation to compare (register, renew, transfer)")
):
    """
    Compare prices across all registrars for a specific TLD and operation
    
    Args:
        tld: The TLD to compare
        operation: The operation type
        
    Returns:
        Comparison of prices across registrars
    """
    tld_clean = tld.lower()
    if not tld_clean.startswith('.'):
        tld_clean = '.' + tld_clean
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT registrar, MIN(price) as price, currency, 
                       pricing_notes, promo_code, source_url
                FROM structured_pricing
                WHERE LOWER(tld) = ? AND LOWER(operation) = ?
                GROUP BY registrar
                ORDER BY price ASC
            """, (tld_clean, operation.lower()))
            
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            
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
    task_id: Optional[str] = Query(None, description="Delete data for specific task"),
    confirm: bool = Query(False, description="Confirmation flag")
):
    """
    Delete pricing data (use with caution!)
    
    Args:
        task_id: Optional task ID to delete specific task data
        confirm: Must be True to proceed with deletion
        
    Returns:
        Deletion status
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Deletion requires confirm=true parameter"
        )
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if task_id:
                # Delete specific task data
                cursor.execute("DELETE FROM structured_pricing WHERE task_id = ?", (task_id,))
                cursor.execute("DELETE FROM raw_data WHERE task_id = ?", (task_id,))
                cursor.execute("DELETE FROM scraping_tasks WHERE task_id = ?", (task_id,))
                deleted_count = cursor.rowcount
                message = f"Deleted data for task {task_id}"
            else:
                # Delete all data
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


# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)