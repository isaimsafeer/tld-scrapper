from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List, Optional
import logging
import uuid
from datetime import datetime
import json
import os
from pathlib import Path

# Import automation function
from automation.automate import run_automation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store automation results
automation_results: Dict[str, Dict[str, Any]] = {}


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
    pricing_notes: str
    promo_code: str
    scraped_at: str
    source_url: str
    column_name: str


class PricingResponse(BaseModel):
    """Response model for pricing data"""
    total_records: int
    filtered_records: int
    data: List[PricingRecord]
    filters_applied: Dict[str, Any]


def load_all_pricing_data() -> List[Dict[str, Any]]:
    """
    Load all pricing data from JSON files in the current directory
    
    Returns:
        List of all pricing records
    """
    all_data = []
    
    # Look for all structured_data_*.json files
    json_files = list(Path(".").glob("structured_data_*.json"))
    
    logger.info(f"Found {len(json_files)} pricing data files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    logger.info(f"Loaded {len(data)} records from {json_file}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    return all_data


def filter_pricing_data(
    data: List[Dict[str, Any]],
    tld: Optional[str] = None,
    registrar: Optional[str] = None,
    operation: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    currency: Optional[str] = None,
    years: Optional[int] = None,
    has_promo: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Filter pricing data based on various criteria
    
    Args:
        data: List of pricing records
        tld: Filter by TLD (e.g., '.com', '.io')
        registrar: Filter by registrar name
        operation: Filter by operation type (register, renew, transfer, multi_year)
        min_price: Minimum price
        max_price: Maximum price
        currency: Filter by currency (USD, GBP, EUR)
        years: Filter by number of years
        has_promo: Filter records with/without promo codes
        
    Returns:
        Filtered list of pricing records
    """
    filtered = data
    
    # Filter by TLD
    if tld:
        tld_lower = tld.lower()
        if not tld_lower.startswith('.'):
            tld_lower = '.' + tld_lower
        filtered = [r for r in filtered if r.get('tld', '').lower() == tld_lower]
    
    # Filter by registrar
    if registrar:
        registrar_lower = registrar.lower()
        filtered = [r for r in filtered if registrar_lower in r.get('registrar', '').lower()]
    
    # Filter by operation
    if operation:
        operation_lower = operation.lower()
        filtered = [r for r in filtered if r.get('operation', '').lower() == operation_lower]
    
    # Filter by price range
    if min_price is not None:
        filtered = [r for r in filtered if r.get('price', 0) >= min_price]
    
    if max_price is not None:
        filtered = [r for r in filtered if r.get('price', float('inf')) <= max_price]
    
    # Filter by currency
    if currency:
        currency_upper = currency.upper()
        filtered = [r for r in filtered if r.get('currency', '').upper() == currency_upper]
    
    # Filter by years
    if years is not None:
        filtered = [r for r in filtered if r.get('years', 0) == years]
    
    # Filter by promo code presence
    if has_promo is not None:
        if has_promo:
            filtered = [r for r in filtered if r.get('promo_code', '').strip() != '']
        else:
            filtered = [r for r in filtered if r.get('promo_code', '').strip() == '']
    
    return filtered


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
    Wrapper function to execute automation and store results
    
    Args:
        task_id: Unique task identifier
        url: Target website URL
        timeout: Maximum wait time for elements
        headless: Run in headless mode
    """
    automation_results[task_id]["status"] = "running"
    
    # Run automation
    result = run_automation(task_id, url, timeout, headless)
    
    # Update results
    automation_results[task_id].update(result)


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
    
    # Initialize task status
    automation_results[task_id] = {
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "url": str(request.url)
    }
    
    # Add automation to background tasks
    background_tasks.add_task(
        execute_automation_task,
        task_id,
        str(request.url),
        request.timeout,
        request.headless
    )
    
    logger.info(f"Task {task_id} queued for {request.url}")
    
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
    if task_id not in automation_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return JSONResponse(content=automation_results[task_id])


@app.get("/pricing", response_model=PricingResponse)
async def get_pricing_data(
    tld: Optional[str] = Query(None, description="Filter by TLD (e.g., 'com', '.com', 'io')"),
    registrar: Optional[str] = Query(None, description="Filter by registrar name (e.g., 'spaceship', 'cloudflare')"),
    operation: Optional[str] = Query(None, description="Filter by operation (register, renew, transfer, multi_year)"),
    min_price: Optional[float] = Query(None, description="Minimum price", ge=0),
    max_price: Optional[float] = Query(None, description="Maximum price", ge=0),
    currency: Optional[str] = Query(None, description="Filter by currency (USD, GBP, EUR)"),
    years: Optional[int] = Query(None, description="Filter by number of years", ge=1),
    has_promo: Optional[bool] = Query(None, description="Filter by promo code availability"),
    sort_by: str = Query('price', description="Sort by field (price, tld, registrar, scraped_at)"),
    sort_order: str = Query('asc', description="Sort order (asc, desc)"),
    limit: Optional[int] = Query(None, description="Limit number of results", ge=1),
    offset: int = Query(0, description="Offset for pagination", ge=0)
):
    """
    Get pricing data with filters and sorting
    
    Query Parameters:
        - tld: Filter by top-level domain (e.g., '.com', 'io')
        - registrar: Filter by registrar name (partial match)
        - operation: Filter by operation type
        - min_price: Minimum price filter
        - max_price: Maximum price filter
        - currency: Filter by currency code
        - years: Filter by registration years
        - has_promo: Filter by promo code availability
        - sort_by: Field to sort by
        - sort_order: Sorting direction (asc/desc)
        - limit: Maximum number of results to return
        - offset: Number of records to skip (for pagination)
    
    Returns:
        Filtered and sorted pricing data
    """
    # Load all pricing data
    all_data = load_all_pricing_data()
    
    if not all_data:
        return PricingResponse(
            total_records=0,
            filtered_records=0,
            data=[],
            filters_applied={}
        )
    
    # Apply filters
    filtered_data = filter_pricing_data(
        all_data,
        tld=tld,
        registrar=registrar,
        operation=operation,
        min_price=min_price,
        max_price=max_price,
        currency=currency,
        years=years,
        has_promo=has_promo
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
    
    Query Parameters:
        - tld: Filter by specific TLD
        - operation: Filter by specific operation
        - currency: Filter by currency
        - group_by: How to group results (tld, operation, registrar)
    
    Returns:
        Cheapest pricing options
    """
    # Load all pricing data
    all_data = load_all_pricing_data()
    
    if not all_data:
        return JSONResponse(content={"message": "No pricing data available", "data": []})
    
    # Apply filters
    filtered_data = filter_pricing_data(
        all_data,
        tld=tld,
        operation=operation,
        currency=currency
    )
    
    if not filtered_data:
        return JSONResponse(content={"message": "No data matching filters", "data": []})
    
    # Group and find cheapest
    grouped_data = {}
    
    for record in filtered_data:
        if group_by == "tld":
            key = (record.get('tld', ''), record.get('operation', ''))
        elif group_by == "operation":
            key = (record.get('operation', ''), record.get('tld', ''))
        elif group_by == "registrar":
            key = (record.get('registrar', ''), record.get('tld', ''))
        else:
            key = record.get('tld', '')
        
        if key not in grouped_data or record.get('price', float('inf')) < grouped_data[key].get('price', float('inf')):
            grouped_data[key] = record
    
    # Convert to list
    result = list(grouped_data.values())
    
    # Sort by price
    result.sort(key=lambda x: x.get('price', 0))
    
    return JSONResponse(content={
        "total_results": len(result),
        "group_by": group_by,
        "data": result
    })


@app.get("/pricing/stats")
async def get_pricing_stats():
    """
    Get statistics about the pricing data
    
    Returns:
        Statistics including available TLDs, registrars, price ranges, etc.
    """
    all_data = load_all_pricing_data()
    
    if not all_data:
        return JSONResponse(content={"message": "No pricing data available"})
    
    # Calculate statistics
    tlds = set(r.get('tld', '') for r in all_data)
    registrars = set(r.get('registrar', '') for r in all_data)
    operations = set(r.get('operation', '') for r in all_data)
    currencies = set(r.get('currency', '') for r in all_data)
    
    prices = [r.get('price', 0) for r in all_data if r.get('price') is not None]
    
    stats = {
        "total_records": len(all_data),
        "unique_tlds": len(tlds),
        "tlds": sorted(list(tlds)),
        "unique_registrars": len(registrars),
        "registrars": sorted(list(registrars)),
        "operations": sorted(list(operations)),
        "currencies": sorted(list(currencies)),
        "price_stats": {
            "min": min(prices) if prices else 0,
            "max": max(prices) if prices else 0,
            "avg": sum(prices) / len(prices) if prices else 0,
            "count": len(prices)
        },
        "records_with_promo": len([r for r in all_data if r.get('promo_code', '').strip() != ''])
    }
    
    return JSONResponse(content=stats)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Selenium Automation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /automate": "Trigger automation task",
            "GET /automation/{task_id}": "Get automation task status",
            "GET /pricing": "Get pricing data with filters",
            "GET /pricing/cheapest": "Get cheapest prices",
            "GET /pricing/stats": "Get pricing statistics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Changed from 8000 to 8080 and added PORT env var
    uvicorn.run(app, host="0.0.0.0", port=port)